__all__ = ['PAFNet_backbone']

from typing import Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_layers import *


class PAFNet_backbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int, 
                 max_seq_len: Optional[int] = 1024, n_layers: int = 3, d_model: int = 128, n_heads: int = 16, 
                 d_k: Optional[int] = None, d_v: Optional[int] = None, d_ff: int = 256, norm: str = 'BatchNorm', 
                 attn_dropout: float = 0., dropout: float = 0., act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True, 
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True, 
                 fc_dropout: float = 0., head_dropout: float = 0, padding_patch = None, pretrain_head: bool = False, 
                 head_type: str = 'flatten', individual: bool = False, revin: bool = False, affine: bool = True, 
                 subtract_last: bool = False, verbose: bool = False, expansion_factor: int = 1, 
                 original_c_in: Optional[int] = None, debug_timing: bool = False, **kwargs):
        super().__init__()
        
        # Patching configuration
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.patch_num = int((context_window - patch_len)/stride + 1)
        self.d_model = d_model
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1
        
        # Backbone encoder
        self.backbone = TSTiEncoder(
            c_in, patch_num=self.patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
            n_layers=n_layers, d_model=self.d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
            attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, 
            padding_var=padding_var, attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, 
            store_attn=store_attn, pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs
        )

        # Head configuration
        self.head_nf = d_model * self.patch_num
        
        if original_c_in is not None:
            real_vars = original_c_in
        else:
            real_vars = c_in // expansion_factor if expansion_factor > 1 else c_in
        
        self.n_vars = real_vars
        self.c_in = c_in
        self.expansion_factor = expansion_factor
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        # Cross-variable attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout, batch_first=True
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, real_vars, fc_dropout)
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
    
    def forward(self, z):
        bs, nvars, seq_len, k_plus_1 = z.shape
        
        if self.padding_patch == 'end':
            z_flat = z.permute(0, 1, 3, 2).reshape(bs, nvars * k_plus_1, seq_len)
            z_patched = self.padding_patch_layer(z_flat)
            z_padded_flat = z_patched.squeeze(1)
            padded_seq_len = z_padded_flat.shape[1]
            z = z_padded_flat.view(bs, nvars, k_plus_1, -1).permute(0, 1, 3, 2)
        
        bs, nvars, seq_len, k_plus_1 = z.shape
        z = z.reshape(bs, nvars * k_plus_1, seq_len)
        z = z.unfold(dimension=2, size=self.patch_len, step=self.stride)
        z = z.reshape(bs, nvars*(k_plus_1), self.patch_len, -1)
        z = self.backbone(z)

        z = z.reshape(bs, nvars, k_plus_1, self.d_model, self.patch_num)
        if k_plus_1 > 1:
            z = self.cross_var_attention(z)
        
        z = self.head(z)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )

    def cross_var_attention(self, z):
        """
        Cross-variable attention with masking to ensure:
        - Each original sequence (variable) only attends to its own leader sequences.
        - No cross-variable or cross-batch attention occurs.
        
        Args:
            z: Tensor of shape [bs, nvars, k_plus_1, d_model, patch_num]
            where k_plus_1 = 1 (original) + K (leaders)
        Returns:
            Tensor of shape [bs, nvars, d_model, patch_num]
        """
        bs, nvars, k_plus_1, d_model, patch_num = z.shape
        
        if k_plus_1 == 1:
            return z.squeeze(2)  # No leaders to attend to
        
        # Separate original and leaders
        z_orig = z[:, :, 0]  # [bs, nvars, d_model, patch_num]
        z_leaders = z[:, :, 1:]  # [bs, nvars, K, d_model, patch_num]
        K = z_leaders.shape[2]
        
        # Prepare queries (original sequences)
        query = z_orig.permute(0, 1, 3, 2).reshape(bs * nvars, patch_num, d_model)  # [bs*nvars, patch_num, d_model]
        
        # Prepare keys/values (leader sequences)
        key_value = z_leaders.permute(0, 1, 4, 2, 3).reshape(bs * nvars, K * patch_num, d_model)  # [bs*nvars, K*patch_num, d_model]
        
        # --- Critical: Create attention mask ---
        # We need a mask of shape [bs*nvars, patch_num, K*patch_num]
        # where each original sequence can ONLY attend to its own leaders
        
        # Method 2: Alternative validation (recommended)
        # Since we've reshaped to [bs*nvars,...], the attention is automatically
        # constrained within each (bs,nvars) group by dimension alignment
        
        # Apply masked multi-head attention
        attended_output, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
            attn_mask=None  # No mask needed due to reshape grouping
        )
        
        # Residual connection with learned weight
        z_combined = query + torch.sigmoid(self.alpha) * attended_output  # [bs*nvars, patch_num, d_model]
        
        # Reshape back to original format
        z_out = z_combined.view(bs, nvars, patch_num, d_model).permute(0, 1, 3, 2)  # [bs, nvars, d_model, patch_num]
        
        return z_out


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
        
    
    
class TSTiEncoder(nn.Module):
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024, n_layers=3, d_model=128, 
                 n_heads=16, d_k=None, d_v=None, d_ff=256, norm='BatchNorm', attn_dropout=0., 
                 dropout=0., act="gelu", store_attn=False, key_padding_mask='auto', padding_var=None, 
                 attn_mask=None, res_attention=True, pre_norm=False, pe='zeros', learn_pe=True, 
                 verbose=False, **kwargs):
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len

        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder = TSTEncoder(
            q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, 
            attn_dropout=attn_dropout, dropout=dropout, pre_norm=pre_norm, 
            activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn
        )

        
    def forward(self, x) -> Tensor:
        n_vars = x.shape[1]
        
        x = x.permute(0,1,3,2)
        x = self.W_P(x)

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))
        u = self.dropout(u + self.W_pos)
        u = u + self.W_pos
        
        z = self.encoder(u)
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))
        z = z.permute(0,1,3,2)

        return z    
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([
            TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, 
                          norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                          activation=activation, res_attention=res_attention,
                          pre_norm=pre_norm, store_attn=store_attn) 
            for i in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: 
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: 
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", 
                 res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, 
                                           proj_dropout=dropout, res_attention=res_attention)

        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias)
        )

        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, 
                attn_mask: Optional[Tensor] = None) -> Tensor:
        if self.pre_norm:
            src = self.norm_attn(src)
        
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        if self.store_attn:
            self.attn = attn
        
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)
        
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., 
                 proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, 
                                                 res_attention=self.res_attention, lsa=lsa)

        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, 
                prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, 
                attn_mask: Optional[Tensor] = None):
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, 
                                                            key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)

        if self.res_attention: 
            return output, attn_weights, attn_scores
        else: 
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None, 
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        attn_scores = torch.matmul(q, k) * self.scale

        if prev is not None: 
            attn_scores = attn_scores + prev

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        if self.res_attention: 
            return output, attn_weights, attn_scores
        else: 
            return output, attn_weights




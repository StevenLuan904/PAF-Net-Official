__all__ = ['PAFNet']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch_dct as dct

from layers.PAFNet_backbone import PAFNet_backbone
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, 
                 d_v: Optional[int] = None, norm: str = 'BatchNorm', attn_dropout: float = 0., 
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None, 
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True, pre_norm: bool = False, 
                 store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True, 
                 pretrain_head: bool = False, head_type: str = 'flatten', verbose: bool = False, **kwargs):
        super().__init__()
        
        # Configuration parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        individual = configs.individual
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        decomposition = configs.decomposition
        
        # PAFNet-specific parameters
        self.decomp_method = getattr(configs, 'decomp_method', 'dct')
        self.k_leaders = getattr(configs, 'patch_leaders', 0)
        self.variable_batch_size = getattr(configs, 'variable_batch_size', 32)
        self.use_leader_integration = getattr(configs, 'use_leader_integration', 1) == 1
        self.corr_threshold = getattr(configs, 'corr_threshold', 0.1)
        self.decomposition = decomposition
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.context_window = context_window
        
        # RevIN normalization
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Learnable integration parameters
        self.base_original_weight = nn.Parameter(torch.tensor(0.6))
        self.base_leader_weight = nn.Parameter(torch.tensor(0.4))
        self.correlation_scale = nn.Parameter(torch.tensor(1.0))
        self.leader_adjustments = nn.Parameter(torch.zeros(self.k_leaders))
        
        # Leader state tracking
        self.last_leader_ids = None
        self.last_shift = None
        self.last_r = None
        
        # Model architecture
        self.expansion_factor = self.k_leaders + 1
        c_in_effective = c_in * self.expansion_factor
        
        self.component_models = nn.ModuleList([
            PAFNet_backbone(
                c_in=c_in_effective, context_window=context_window, target_window=target_window, 
                patch_len=patch_len, stride=stride, max_seq_len=max_seq_len,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                dropout=dropout, act=act, key_padding_mask=key_padding_mask, 
                padding_var=padding_var, attn_mask=attn_mask, 
                res_attention=res_attention, pre_norm=pre_norm, 
                store_attn=store_attn, pe=pe, learn_pe=learn_pe, 
                fc_dropout=fc_dropout, head_dropout=head_dropout, 
                padding_patch=padding_patch, pretrain_head=pretrain_head, 
                head_type=head_type, individual=individual, 
                revin=False, affine=affine, subtract_last=subtract_last, 
                verbose=verbose, expansion_factor=self.expansion_factor,
                original_c_in=c_in, **kwargs
            ) for _ in range(self.decomposition)
        ])
    
    
    def forward(self, x):
        if self.revin:
            x = self.revin_layer(x, 'norm')

        x = x.permute(0, 2, 1)
        x_expanded = self._expand_with_leaders(x)
        x_decomposed = self._decompose_frequencies(x_expanded)
        
        outputs = []
        for cur_f in range(self.decomposition):
            component_input = x_decomposed[:, :, :, :, cur_f]
            component_output = self.component_models[cur_f](component_input)
            outputs.append(component_output)
        
        x_out = torch.stack(outputs, dim=-1).sum(dim=-1)
        x_out = x_out.permute(0, 2, 1)

        if self.revin:
            x_out = self.revin_layer(x_out, 'denorm')

        return x_out
    
    def _expand_with_leaders(self, x, leader_ids=None, shift=None, r=None, mode="phase"):
        B, C, L = x.shape
        device = x.device
        
        if leader_ids is None or shift is None or r is None:
            with torch.no_grad():
                leader_ids, shift, r = self.get_leader_info_phas(x)
        
        x_expanded = torch.zeros(B, C, L, self.k_leaders + 1, device=device)
        x_expanded[:, :, :, 0] = x
        
        base_indices = torch.arange(L, device=device).view(1, 1, L, 1)
        shift_expanded = shift.unsqueeze(2)
        shifted_indices = torch.clamp(base_indices - shift_expanded, 0, L - 1)
        
        leader_ids_expanded = leader_ids.unsqueeze(2).expand(B, C, L, self.k_leaders)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, C, L, self.k_leaders)
        
        leader_sequences = x[batch_idx, leader_ids_expanded, shifted_indices]
        
        r_expanded = r.unsqueeze(2)
        correlation_mask = (r_expanded < 0).float()
        leader_sequences = leader_sequences * (1 - 2 * correlation_mask)
        
        r_abs_expanded = r.abs().unsqueeze(2)
        threshold_mask = r_abs_expanded < self.corr_threshold
        original_repeated = x.unsqueeze(-1).expand(-1, -1, -1, self.k_leaders)
        leader_sequences = torch.where(threshold_mask, original_repeated, leader_sequences)

        x_expanded[:, :, :, 1:] = leader_sequences
        return x_expanded

    def phase_correlation(self, x, variable_batch_size=32, eps=1e-8):
        B, C, L = x.shape
        device = x.device
        K = self.k_leaders
        
        rfft = torch.fft.rfft(x, dim=-1)
        mag = torch.abs(rfft).clamp(min=eps)
        norm_rfft = rfft / mag
        
        all_correlations = []
        all_leaders = []
        
        for i in range(0, C, variable_batch_size):
            batch_size = min(variable_batch_size, C - i)
            batch_leaders = slice(i, i + batch_size)
            
            cross_phase = norm_rfft.unsqueeze(2) * torch.conj(norm_rfft[:, batch_leaders]).unsqueeze(1)
            pc_batch = torch.fft.irfft(cross_phase, n=L, dim=-1).real
            
            max_val, max_idx = pc_batch.max(dim=-1)
            max_idx = torch.where(max_idx > L//2, max_idx - L, max_idx)
            
            all_correlations.append((max_val, max_idx))
            all_leaders.append(torch.arange(i, i + batch_size, device=device))
        
        all_max_val = torch.cat([c[0] for c in all_correlations], dim=2)
        all_max_idx = torch.cat([c[1] for c in all_correlations], dim=2)
        all_leader_ids = torch.cat(all_leaders).expand(B, C, C)
        
        self_idx = torch.arange(C, device=device)
        identity_mask = torch.zeros(C, C, device=device)
        identity_mask[self_idx, self_idx] = 1
        identity_mask = identity_mask.expand(B, C, C)
        masked_max_val = all_max_val - 2 * identity_mask
        
        topk_val, topk_idx = masked_max_val.topk(k=K, dim=2)
        
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, C, K)
        var_indices = torch.arange(C, device=device).view(1, C, 1).expand(B, C, K)
        
        shift_steps = all_max_idx[batch_indices, var_indices, topk_idx]
        max_similarity = topk_val
        leader_ids = all_leader_ids[batch_indices, var_indices, topk_idx]
        
        return leader_ids, shift_steps, max_similarity
    
    def get_leader_info_phas(self, x):
        B, C, L = x.shape
        leader_ids, shift, r = self.phase_correlation(x, variable_batch_size=self.variable_batch_size)
        return leader_ids, shift, r

    def _decompose_frequencies(self, x):
        x = x.permute(0, 1, 3, 2)
        B, C, K, L = x.shape
        
        X = dct.dct(x, norm='ortho')
        F = self.decomposition
        masks = torch.zeros(B, C, K, L, F, device=x.device)

        split_points = torch.linspace(0, L, F + 1, dtype=torch.long)
        for i in range(F):
            start, end = split_points[i], split_points[i + 1]
            masks[..., start:end, i] = 1

        bands = dct.idct(X.unsqueeze(-1) * masks, norm='ortho')
        bands = bands.permute(0, 1, 3, 2, 4)
        return bands
    
    def count_parameters(self):
        """Print the number of learnable parameters in the model."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total learnable parameters: {total_params:,}')
        return total_params
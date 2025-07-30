import math
import time
import torch

# Global cache for leader indicators
_global_leader_cache = {}
_cache_hits = 0
_cache_misses = 0


def cross_corr_coef(x, variable_batch_size=32, predefined_leaders=None, local_max=True):
    B, C, L = x.shape

    rfft = torch.fft.rfft(x, dim=-1)  # [B, C, F]
    rfft_conj = torch.conj(rfft)
    if predefined_leaders is None:
        # Add progress bar for the computationally intensive cross-correlation calculation
        batched_results = []
        for i in range(0, C, variable_batch_size):
            batch_result = torch.fft.irfft(
                rfft.unsqueeze(2) * rfft_conj[:, i: i + variable_batch_size].unsqueeze(1),
                dim=-1, n=L
            )
            batched_results.append(batch_result)
        cross_corr = torch.cat(batched_results, 2)  # [B, C, C, L]
    else:
        cross_corr = torch.fft.irfft(
            rfft.unsqueeze(2) * rfft_conj[:, predefined_leaders.view(-1)].view(B, C, -1, rfft.shape[-1]),
            dim=-1, n=L)

    if local_max:
        corr_abs = cross_corr.abs()
        mask = (corr_abs[..., 1:-1] >= corr_abs[..., :-2]) & (corr_abs[..., 1:-1] >= corr_abs[..., 2:])
        cross_corr = cross_corr[..., 1:-1] * mask

    # cross_corr[..., 0] = cross_corr[..., 0] * (1 - torch.eye(cross_corr.shape[1], device=cross_corr.device))
    return cross_corr / L


def estimate_indicator(x, K, variable_batch_size=32, predefined_leaders=None, local_max=True, use_cache=False):
    """
    Calculate leader indicators with optional caching for performance.
    
    Args:
        x: Input tensor [B, C, L]
        K: Number of leaders
        variable_batch_size: Batch size for variable processing
        predefined_leaders: Predefined leader indices if available
        local_max: Whether to use local maxima for correlation
        use_cache: Whether to use the global cache
        
    Returns:
        leader_ids, shift, r: Leader information tensors
    """
    global _global_leader_cache, _cache_hits, _cache_misses
    
    # Generate a cache key if using cache
    if use_cache:
        # More efficient cache key generation using tensor properties
        with torch.no_grad():
            try:
                # Try to generate a hash from the first batch item
                cache_key = (hash(x[0].cpu().numpy().tobytes()), K, local_max)
                
                # Check if we already calculated this
                if cache_key in _global_leader_cache:
                    _cache_hits += 1
                    return _global_leader_cache[cache_key]
            except:
                # Fallback if hashing fails
                use_cache = False
    
    # Compute cross-correlation once
    cross_corr = cross_corr_coef(x, variable_batch_size, predefined_leaders)
    
    # Efficient correlation analysis
    L = x.shape[-1]
    max_shift = L // 3
    
    # Only compute abs once and slice efficiently
    corr_abs = cross_corr[..., :max_shift].abs()  # Combined operation
    corr_abs_max, shift = corr_abs.max(-1)  # [B, C, C]
    
    if not local_max:
        # In-place operation to save memory
        corr_abs_max.mul_(shift > 0)
        
    # More efficient leader selection
    _, leader_ids = corr_abs_max.topk(K, dim=-1)  # [B, C, K]
    
    # Efficient gathering using pre-computed indices
    leader_ids_expanded = leader_ids.unsqueeze(-1).expand(-1, -1, -1, cross_corr.shape[-1])
    lead_corr = cross_corr.gather(2, leader_ids_expanded)  # [B, C, K, L]
    
    shift = shift.gather(2, leader_ids)  # [B, C, K]
    r = lead_corr.gather(3, shift.unsqueeze(-1)).squeeze(-1)  # [B, C, K]
    
    if local_max:
        shift = shift + 1
        
    if predefined_leaders is not None:
        leader_ids = predefined_leaders.unsqueeze(0).expand(len(x), -1, -1).gather(-1, leader_ids)
    
    # Store in cache if using cache
    if use_cache:
        _cache_misses += 1
        
        # More efficient cache management with LRU-like behavior
        if len(_global_leader_cache) >= 300:
            # Remove multiple old entries at once to reduce frequent cleanup
            keys_to_remove = list(_global_leader_cache.keys())[:50]
            for key in keys_to_remove:
                _global_leader_cache.pop(key, None)
        
        _global_leader_cache[cache_key] = (leader_ids, shift, r)
        
        # Limit cache size to avoid memory issues
        if len(_global_leader_cache) > 300:
            # Remove the oldest entry (simple approach)
            _global_leader_cache.pop(next(iter(_global_leader_cache)))
    
    return leader_ids, shift, r


def shifted_leader_seq(x, y_hat, leader_num, leader_ids=None, shift=None, r=None, const_indices=None,
                       variable_batch_size=32, predefined_leaders=None, corr_threshold=0.1):
    B, C, L = x.shape
    H = y_hat.shape[-1]
    
    # Pre-compute indices if not provided
    if const_indices is None:
        const_indices = torch.arange(L, L + H, dtype=torch.int, device=x.device).unsqueeze(0).unsqueeze(0)

    if leader_ids is None:
        leader_ids, shift, r = estimate_indicator(x, leader_num,
                                                  variable_batch_size=variable_batch_size,
                                                  predefined_leaders=predefined_leaders)
    indices = const_indices - shift.view(B, -1, 1)  # [B, C*K, H]

    seq = torch.cat([x, y_hat], -1)  # [B, C, L+H]
    seq = seq.gather(1, leader_ids.view(B, -1, 1).expand(-1, -1, L + H))  # [B, C*K, L+H]
    seq_shifted = seq.gather(-1, indices)
    seq_shifted = seq_shifted.view(B, C, -1, indices.shape[-1])  # [B, C, K, H]

    r = r.view(B, C, -1)  # [B, C, K]
    seq_shifted = seq_shifted * torch.sign(r).unsqueeze(-1)
    
    # Apply correlation threshold: if abs(r) < threshold, replace with original target sequence
    r_abs = r.abs()
    threshold_mask = r_abs < corr_threshold  # [B, C, K]
    
    # Use in-place operations where possible
    if threshold_mask.any():
        original_target = y_hat.unsqueeze(2).expand(-1, -1, leader_num, -1)  # [B, C, K, H]
        seq_shifted = torch.where(threshold_mask.unsqueeze(-1), original_target, seq_shifted)

    return seq_shifted, r_abs

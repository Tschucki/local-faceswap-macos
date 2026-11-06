# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for Local FaceSwap macOS: Apple Silicon (MPS) support and pipeline changes.
# MPS port: flash-attn is CUDA-only. Without it, every call (including the
# direct `flash_attention` calls in model_animate.py) runs through PyTorch SDPA,
# honouring q_lens/k_lens via a boolean mask instead of silently dropping them.
import torch
import torch.nn.functional as F

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

__all__ = [
    'flash_attention',
    'attention',
]


def _sdpa(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
          q_scale=None, causal=False, dtype=None):
    """q/k/v: [B, L, N, C] -> [B, Lq, N, C]. Runs in the autocast dtype if active."""
    out_dtype = q.dtype
    if dtype is None:
        device_type = q.device.type
        if device_type in ('mps', 'cuda') and torch.is_autocast_enabled(device_type):
            dtype = torch.get_autocast_dtype(device_type)
        else:
            dtype = q.dtype
    b, lq, lk = q.size(0), q.size(1), k.size(1)

    attn_mask = None
    if k_lens is not None and int(k_lens.min()) < lk:
        # Boolean key-padding mask (True = attend). Only built when keys are
        # really padded: any mask forces MPS off its fused SDPA kernel (~2.5x
        # slower). Padded query rows need no mask; the caller drops them.
        k_valid = torch.arange(lk, device=q.device)[None] < k_lens.to(q.device)[:, None]
        attn_mask = k_valid[:, None, None, :].expand(b, 1, lq, lk)

    # contiguous [B, N, L, C]: the MPS SDPA kernel is ~13% faster than on views
    q = q.transpose(1, 2).to(dtype).contiguous()
    k = k.transpose(1, 2).to(dtype).contiguous()
    v = v.transpose(1, 2).to(dtype).contiguous()
    if q_scale is not None:
        q = q * q_scale

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=causal and attn_mask is None,
        dropout_p=dropout_p, scale=softmax_scale)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    use_flash = (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE) and q.device.type == 'cuda'
    if not use_flash:
        assert tuple(window_size) == (-1, -1), 'sliding-window attention needs flash-attn'
        return _sdpa(q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
                     softmax_scale=softmax_scale, q_scale=q_scale, causal=causal)

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    return flash_attention(
        q=q, k=k, v=v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
        softmax_scale=softmax_scale, q_scale=q_scale, causal=causal,
        window_size=window_size, deterministic=deterministic, dtype=dtype,
        version=fa_version)

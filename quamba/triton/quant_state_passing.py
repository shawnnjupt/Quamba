"""We want triton==2.1.0 or 2.2.0 for this
"""

import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from einops import rearrange, repeat

from mamba_ssm.ops.triton.softplus import softplus
from quamba.fxp_units import get_exp_tl, get_softplus_tl
_exp_tl = get_exp_tl()
_softplus_tl = get_softplus_tl()



@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['dim'],
)
@triton.jit
def _quant_state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr, out_ptr, final_states_ptr, dA_cs_ptr, initstates_ptr, seq_idx_ptr,
    # Matrix dimensions
    dim, nchunks, seqlen, chunk_size,
    # Strides
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_dim,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_final_states_batch, stride_final_states_head, stride_final_states_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_initstates_batch, stride_initstates_head, stride_initstates_dim,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D_STATE: tl.constexpr, # 传入 dstate 作为常量
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    final_states_ptr += pid_b * stride_final_states_batch + pid_h * stride_final_states_head
    if HAS_INITSTATES:
        initstates_ptr += pid_b * stride_initstates_batch + pid_h * stride_initstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim

    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    else:
        initstates_ptrs = initstates_ptr + offs_m * stride_initstates_dim
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    tl.store(out_ptrs, states, mask=offs_m < dim)
    out_ptrs += stride_out_chunk
    seq_idx = 0

    NUM_ROWS_PER_BLOCK = BLOCK_SIZE // D_STATE

    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        # scale = tl.exp(dA_cs)
        scale =_exp_tl(dA_cs)
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(seq_idx_ptr + (min((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen)
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
            seq_idx = seq_idx_new
        states = scale * states + new_states


        # # mode1 Per-headdim (Per-row) Po2 量化 ---
        # # 直接在 reshape 中写计算公式，不要先赋值给 NUM_ROWS_PER_BLOCK 变量
        # # 这样编译器能确保这两个值都是 constexpr
        # states_2d = tl.reshape(states, [BLOCK_SIZE // D_STATE, D_STATE])
        
        # # 计算每一行的最大绝对值
        # row_max = tl.max(tl.abs(states_2d), axis=1) 
        
        # # 计算 Po2 Scale
        # safe_max = tl.maximum(row_max, 1e-8)
        # p = libdevice.ceil(libdevice.log2(safe_max / 127.0))
        # po2_scale = libdevice.exp2(p) 
        
        # # 广播并量化
        # po2_scale_2d = po2_scale[:, None]
        # q_states = libdevice.round(states_2d / po2_scale_2d)
        
        # # Clamp
        # q_states = tl.where(q_states > 127, 127, q_states)
        # q_states = tl.where(q_states < -128, -128, q_states)
        
        # # 反量化并还原形状
        # states_dequant = q_states * po2_scale_2d
        # # 同样，还原形状时也直接写 BLOCK_SIZE
        # states = tl.reshape(states_dequant, [BLOCK_SIZE])
        # # ---------------------------------------


        # mode2. 计算整个 block 的最大绝对值
        # states 的形状是 (BLOCK_SIZE,)
        max_val = tl.max(tl.abs(states), axis=0)   
        
        # 2. 计算 Po2 Scale
        safe_max = tl.maximum(max_val, 1e-8)   
        p = libdevice.ceil(libdevice.log2(safe_max / 127.0))   
        po2_scale = libdevice.exp2(p)   

        # -----------------------------------------------------------

        # 3. 量化：整个向量使用同一个标量 po2_scale
        q_states = libdevice.round(states / po2_scale)   
        
        # 4. Clamp 到 Int8 范围
        q_states = tl.where(q_states > 127, 127, q_states)   
        q_states = tl.where(q_states < -128, -128, q_states)   
        
        # 5. 反量化：还原回 float32 供下一个 chunk 递推
        states = q_states * po2_scale   
        # ---------------------------------------


        # # mode3. Per-element Po2 量化 ---   
        # # 每个元素独立计算自己的 scale
        
        # # 1. 获取每个元素的绝对值
        # abs_states = tl.abs(states)
        
        # # 2. 为每个元素计算专属的 Po2 Scale
        # # p = ceil(log2(|x| / 127))
        # safe_abs = tl.maximum(abs_states, 1e-8)   
        # p = libdevice.ceil(libdevice.log2(safe_abs / 127.0))   
        # po2_scale = libdevice.exp2(p)   # 这里的 po2_scale 形状也是 (BLOCK_SIZE,)
        
        # # 3. 逐元素量化
        # # 每个元素除以自己专属的 scale，结果理论上都在 [-127, 127] 之间
        # q_states = libdevice.round(states / po2_scale)   
        
        # # 4. Clamp 到 Int8 范围
        # q_states = tl.where(q_states > 127, 127, q_states)   
        # q_states = tl.where(q_states < -128, -128, q_states)   
        
        # 5. 反量化：还原回 float32
        # states = q_states * po2_scale   
        # ---------------------------------------


        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def _quant_state_passing_fwd(states, dA_chunk_cumsum,d_state, initial_states=None, seq_idx=None, chunk_size=None,
                       out_dtype=None):
    batch, nchunks, nheads, dim = states.shape # dim = headdim * dstate

    assert (d_state & (d_state - 1)) == 0, "d_state must be a power of 2"

    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, dim)
    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((batch, nchunks, nheads, dim), device=states.device, dtype=out_dtype)
    final_states = torch.empty((batch, nheads, dim), device=states.device, dtype=out_dtype)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(states.device.index):
        _quant_state_passing_fwd_kernel[grid](
            states, out, final_states, dA_chunk_cumsum, initial_states, seq_idx,
            dim, nchunks, seqlen if seq_idx is not None else 0, chunk_size if seq_idx is not None else 0,
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            final_states.stride(0), final_states.stride(1), final_states.stride(2),
            dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1),
            *((initial_states.stride(0), initial_states.stride(1), initial_states.stride(2))
              if initial_states is not None else (0, 0, 0)),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            D_STATE=d_state, # 必须作为 tl.constexpr 传入
        )
    return out, final_states
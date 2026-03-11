import torch
from einops import rearrange, repeat

from quamba.triton.quant_chunk_cumsum import _quant_chunk_cumsum_fwd
from quamba.triton.quant_state_passing import _quant_state_passing_fwd
from quamba.triton.quant_chunk_state import _quant_chunk_state_fwd, _quamba2_chunk_state_fwd
from quamba.triton.quant_chunk_scan import _quant_chunk_scan_fwd, _quamba2_chunk_scan_fwd
from quamba.triton.quant_bmm_chunk import _quant_bmm_chunk_fwd, _quamba2_bmm_chunk_fwd
from quamba.triton.quant_ssm_states import _quant_quant_ssm_states, _quamba2_quant_ssm_states

import os
import numpy as np
_DUMP_DIR = "/deltadisk/congxiao/code/github/Quamba/dump"

def _dump_fp32_txt(name, tensor):
    os.makedirs(_DUMP_DIR, exist_ok=True)
    path = os.path.join(_DUMP_DIR, f"{name}.txt")

    if tensor is None:
        with open(path, "w") as f:
            f.write("None\n")
        return

    if not torch.is_tensor(tensor):
        with open(path, "w") as f:
            f.write(str(tensor) + "\n")
        return

    t = tensor.detach().contiguous().float().cpu()
    flat = t.view(-1)

    with open(path, "w") as f:
        for v in flat:
            f.write(f"{v.item():.8f}\n")

def _select_first_head(t, head_dim_size=24):
    """
    自动寻找 size == head_dim_size 的维度，并选取 head=0
    """
    if not torch.is_tensor(t):
        return t

    for dim, size in enumerate(t.shape):
        if size == head_dim_size:
            return t.select(dim=dim, index=0)

    # 没找到 head 维度，直接返回
    return t

def _dump_bin(name, tensor):
    os.makedirs(_DUMP_DIR, exist_ok=True)
    bin_path = os.path.join(_DUMP_DIR, f"{name}.bin")
    txt_path = os.path.join(_DUMP_DIR, f"{name}.txt")

    # ---------------- None ----------------
    if tensor is None:
        open(bin_path, "wb").close()
        open(txt_path, "w").write("None\n")
        return

    # ---------------- 非 tensor ----------------
    if not torch.is_tensor(tensor):
        arr = np.array([tensor], dtype=np.float32)
        arr.tofile(bin_path)
        with open(txt_path, "w") as f:
            f.write(f"{float(arr[0]):.8f}\n")
        return

    # ---------------- tensor ----------------
    t = tensor.detach()
    # 只取第一个 head
    t = _select_first_head(t)
    t = t.contiguous()
    # int8 原样 dump
    if t.dtype == torch.int8:
        arr = t.cpu().numpy().astype(np.int8)
    else:
        arr = t.float().cpu().numpy().astype(np.float32)
    # bin
    arr.tofile(bin_path)
    # txt
    flat = arr.reshape(-1)
    with open(txt_path, "w") as f:
        for v in flat:
            f.write(f"{int(v)}\n" if arr.dtype == np.int8 else f"{v:.8f}\n")

def dump_ssm_scale_bin(name, tensor):
    """
    Dump SSM scale tensor (power-of-two).

    Outputs:
        name.bin        -> int32 exponent k
        name.txt        -> int exponent (one per line)
        name_fp32.txt   -> original fp32 values
    """
    import os
    import numpy as np
    import torch

    os.makedirs(_DUMP_DIR, exist_ok=True)

    bin_path      = os.path.join(_DUMP_DIR, f"{name}.bin")
    txt_int_path  = os.path.join(_DUMP_DIR, f"{name}.txt")
    txt_fp32_path = os.path.join(_DUMP_DIR, f"{name}_fp32.txt")

    # -------- None --------
    if tensor is None:
        open(bin_path, "wb").close()
        with open(txt_int_path, "w") as f:
            f.write("None\n")
        with open(txt_fp32_path, "w") as f:
            f.write("None\n")
        return

    if not torch.is_tensor(tensor):
        raise TypeError("dump_ssm_scale_bin expects a torch.Tensor")

    # -------- tensor --------
    t = tensor.detach().contiguous()

    # flatten & take first 4 x 128
    t = t.view(-1)[: 4 * 128]

    # fp32 values
    t_fp32 = t.float().cpu().numpy()

    # fp32 -> exponent int (value = 2^k)
    k_int = np.round(np.log2(t_fp32)).astype(np.int32)

    # -------- dump bin (int32) --------
    k_int.tofile(bin_path)

    # -------- dump int txt --------
    with open(txt_int_path, "w") as f:
        for v in k_int:
            f.write(f"{int(v)}\n")

    # -------- dump fp32 txt --------
    with open(txt_fp32_path, "w") as f:
        for v in t_fp32:
            f.write(f"{v:.8f}\n")


def _quant_mamba_chunk_scan_combined_fwd(
        q_x, x_scale, q_dt, dt_scale, q_A_log, A_log_scale,
        q_B, B_scale, q_C, C_scale, ssm_state_scale, chunk_size,
        q_D=None, D_scale=None, q_z=None, z_scale=None, dt_bias=None, initial_states=None, seq_idx=None,
        cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), mm_dtype=torch.float16
    ):
    _, _, ngroups, dstate = q_B.shape
    batch, seqlen, nheads, headdim = q_x.shape

    assert x_scale.is_cuda
    assert x_scale.numel() == 1
    assert B_scale.is_cuda
    assert B_scale.numel() == 1
    assert C_scale.is_cuda
    assert C_scale.numel() == 1

    assert nheads % ngroups == 0
    assert q_x.is_cuda
    assert q_x.dtype == torch.int8
    assert q_x.shape == (batch, seqlen, nheads, headdim)
    assert q_B.is_cuda
    assert q_B.dtype == torch.int8
    assert q_B.shape == (batch, seqlen, ngroups, dstate)
    assert q_dt.is_cuda
    assert q_dt.dtype == torch.int8
    assert q_dt.shape == (batch, seqlen, nheads)
    assert q_A_log.is_cuda
    assert q_A_log.dtype == torch.int8
    assert q_A_log.shape == (nheads,)
    assert q_C.is_cuda
    assert q_C.dtype == torch.int8
    assert q_C.shape == q_B.shape
    if q_z is not None:
        assert q_z.shape == q_x.shape
    if q_D is not None:
        assert q_D.shape == (nheads, headdim) or q_D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if q_B.stride(-1) != 1:
        q_B = q_B.contiguous()
    if q_C.stride(-1) != 1:
        q_C = q_C.contiguous()
    if q_x.stride(-1) != 1 and q_x.stride(1) != 1:  # Either M or K dimension should be contiguous
        q_x = q_x.contiguous()
    if q_z is not None and q_z.stride(-1) != 1 and q_z.stride(1) != 1:  # Either M or K dimension should be contiguous
        q_z = q_z.contiguous()
    if q_D is not None and q_D.stride(-1) != 1:
        q_D = q_D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    dA_cumsum, dt = _quant_chunk_cumsum_fwd(q_dt, dt_scale, q_A_log, A_log_scale, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states = _quant_chunk_state_fwd(q_B, B_scale, q_x, x_scale, dt, dA_cumsum, mm_dtype=torch.float16, seq_idx=seq_idx, states_in_fp32=True)
    states, final_states = _quant_state_passing_fwd(
                                rearrange(states, "... p n -> ... (p n)"),
                                dA_cumsum[:, :, :, -1],
                                initial_states=rearrange(initial_states, "... p n -> ... (p n)") \
                                    if initial_states is not None else None,
                                seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=mm_dtype
                            )
    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]
    CB = _quant_bmm_chunk_fwd(q_C, C_scale, q_B, B_scale, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    out, out_x = _quant_chunk_scan_fwd(
        CB, q_x, x_scale, dt, dA_cumsum, q_C, C_scale, states,
        q_D=q_D, D_scale=D_scale, q_z=q_z, z_scale=z_scale,
        seq_idx=seq_idx, mm_dtype=torch.float16
    )
    final_states = _quant_quant_ssm_states(final_states, ssm_state_scale)
    if cu_seqlens is None:
        return out, final_states
    else:
        raise NotImplementedError("Only supports `cu_seqlens=None`")


def _quamba2_mamba_chunk_scan_combined_fwd(
        q_x, x_scales, x_head_group_range, x_dim_group_range,
        q_dt, dt_scale, q_A_log, A_log_scale, q_B, B_scale, q_C, C_scale, ssm_state_scale, chunk_size,
        q_D=None, D_scale=None, q_z=None, z_scale=None, dt_bias=None, initial_states=None, seq_idx=None,
        cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), mm_dtype=torch.float16
    ):
    _, _, ngroups, dstate = q_B.shape
    batch, seqlen, nheads, headdim = q_x.shape
    assert len(x_head_group_range.shape) == 2, "x_head_group_range must have shape [n_ssd_group, x_nhead_group]"
    assert len(x_dim_group_range.shape) == 3, "x_dim_group_range must have shape [n_ssd_group, x_nhead_group, n_dim_group]"
    nhead_groups = x_head_group_range.shape[1] # [n_ssd_groups, n_head_groups]
    ndim_groups = x_dim_group_range.shape[2] # [n_ssd_groups, n_head_groups, n_dim_groups]
    assert x_scales.is_cuda
    assert x_head_group_range.is_cuda
    assert x_dim_group_range.is_cuda
    assert x_scales.numel() == ngroups*nhead_groups*ndim_groups, \
            f"{x_scales.numel()} vs. {ngroups}*{nhead_groups}*{ndim_groups}"
    assert x_head_group_range.dtype == torch.int32
    assert x_dim_group_range.dtype == torch.int32

    assert B_scale.is_cuda
    assert B_scale.numel() == ngroups
    assert C_scale.is_cuda
    assert C_scale.numel() == ngroups

    assert nheads % ngroups == 0
    assert q_x.is_cuda
    assert q_x.dtype == torch.int8
    assert q_x.shape == (batch, seqlen, nheads, headdim)
    assert q_B.is_cuda
    assert q_B.dtype == torch.int8
    assert q_B.shape == (batch, seqlen, ngroups, dstate)
    assert q_dt.is_cuda
    assert q_dt.dtype == torch.int8
    assert q_dt.shape == (batch, seqlen, nheads)
    assert q_A_log.is_cuda
    assert q_A_log.dtype == torch.int8
    assert q_A_log.shape == (nheads,)
    assert q_C.is_cuda
    assert q_C.dtype == torch.int8
    assert q_C.shape == q_B.shape
    assert ssm_state_scale.is_cuda
    assert ssm_state_scale.dtype == torch.float32
    assert ssm_state_scale.shape == (ngroups, nhead_groups, ndim_groups, dstate)
    if q_z is not None:
        assert q_z.shape == q_x.shape
    if q_D is not None:
        assert q_D.shape == (nheads, headdim) or q_D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if q_B.stride(-1) != 1:
        q_B = q_B.contiguous()
    if q_C.stride(-1) != 1:
        q_C = q_C.contiguous()
    if q_x.stride(-1) != 1 and q_x.stride(1) != 1:  # Either M or K dimension should be contiguous
        q_x = q_x.contiguous()
    if q_z is not None and q_z.stride(-1) != 1 and q_z.stride(1) != 1:  # Either M or K dimension should be contiguous
        q_z = q_z.contiguous()
    if q_D is not None and q_D.stride(-1) != 1:
        q_D = q_D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)

# ---------------- DUMP INPUT (ONLY ONCE) ----------------
    # if not hasattr(_quant_mamba_chunk_scan_combined_fwd, "_dump_done"):
    #     _quant_mamba_chunk_scan_combined_fwd._dump_input_this_call = True
    #     _dump_bin("_quant_chunk_cumsum_fwd_input_q_dt", q_dt)
    #     _dump_bin("_quant_chunk_cumsum_fwd_input_dt_scale", dt_scale)
    #     _dump_bin("_quant_chunk_cumsum_fwd_input_q_A_log", q_A_log)
    #     _dump_bin("_quant_chunk_cumsum_fwd_input_A_log_scale", A_log_scale)

    #     _dump_bin("_quant_chunk_cumsum_fwd_input_chunk_size", chunk_size)
    #     _dump_bin("_quant_chunk_cumsum_fwd_input_dt_softplus", dt_softplus)
    #     _dump_bin("_quant_chunk_cumsum_fwd_input_dt_limit_min", dt_limit[0])
    #     _dump_bin("_quant_chunk_cumsum_fwd_input_dt_limit_max", dt_limit[1])

    #     if dt_bias is not None:
    #         _dump_bin("_quant_chunk_cumsum_fwd_input_dt_bias", dt_bias)
    #     else:
    #         _dump_bin("_quant_chunk_cumsum_fwd_input_dt_bias", None)
    #     # print(f"dt_scale={dt_scale}")
    #     # print(f"q_dt={q_dt}")
    #     # print(f"dt_bias={dt_bias}")
    # else:
    #     _quant_mamba_chunk_scan_combined_fwd._dump_input_this_call = False


    # -------------------------------------------------------------------------
    dA_cumsum, dt = _quant_chunk_cumsum_fwd(q_dt, dt_scale, q_A_log, A_log_scale, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    # ---------------- DUMP OUTPUT (MATCH INPUT) ----------------
    # if getattr(_quant_mamba_chunk_scan_combined_fwd, "_dump_input_this_call", False):
    #     _dump_bin("_quant_chunk_cumsum_fwd_output_dA_cumsum", dA_cumsum)
    #     _dump_bin("_quant_chunk_cumsum_fwd_output_dt", dt)
    #     print(f"out_dt={dt[0,0,:,:]}")
    #     print(f"out_dt_shape={dt.shape}")
    #     torch.set_printoptions(threshold=float('inf'))
    #     print(f"output_cumsum={dA_cumsum[0,0,:,:]}")
    #     print(f"dA_cumsum={dA_cumsum.shape}")
    #     _quant_mamba_chunk_scan_combined_fwd._dump_done = True


    # if getattr(_quant_mamba_chunk_scan_combined_fwd, "_dump_input_this_call", False):
    #     _dump_bin("_quamba2_chunk_state_fwd_q_B", q_B)
    #     _dump_bin("_quamba2_chunk_state_fwd_B_scale", B_scale)
    #     _dump_bin("_quamba2_chunk_state_fwd_q_x", q_x)
    #     _dump_bin("_quamba2_chunk_state_fwd_x_scales", x_scales)
    #     _dump_bin("_quamba2_chunk_state_fwd_x_head_group_range", x_head_group_range)
    #     _dump_bin("_quamba2_chunk_state_fwd_x_dim_group_range", x_dim_group_range)
    #     _dump_bin("_quamba2_chunk_state_fwd_x_head_group_range", x_head_group_range)
    #     _dump_bin("_quamba2_chunk_state_fwd_dt", dt)
    #     _dump_bin("_quamba2_chunk_state_fwd_dA_cumsum", dA_cumsum)
    #     print(f"q_B={q_B}")
    #     print(f"q_B_shape={q_B.shape}")
    #     print(f"q_x={q_x}")
    #     print(f"q_x_shape={q_x.shape}")
    #     print(f"q_c={q_C}")
    #     print(f"q_c_shape={q_C.shape}")
    states = _quamba2_chunk_state_fwd(q_B, B_scale, q_x, x_scales, x_head_group_range, x_dim_group_range, dt, dA_cumsum, mm_dtype=torch.float16, seq_idx=seq_idx, states_in_fp32=True)
    # if getattr(_quant_mamba_chunk_scan_combined_fwd, "_dump_input_this_call", False):
        # print(f"q_B={q_B}")
        # print(f"q_B_shape={q_B.shape}")
        # print(f"q_B_scale={B_scale}")
        # print(f"q_x_scales={x_scales}")
        # print(f"x_head_group_range={x_head_group_range}")
        # print(f"x_dim_group_range={x_dim_group_range}")
        # print(f"q_x={q_x[0,:,0,:]}")
        # print(f"q_x_shape={q_x.shape}")
        # print(f"pre_states={states[0, :, 0, :,:] }")
        # print(f"pre_states_shape={states.shape}")
    d_state_val = states.shape[-1]
    # print("initial_states=", initial_states.shape if initial_states is not None else "None")
    # print("initial_states_dtype=",initial_states.dtype if initial_states is not None else "None")
    states, final_states = _quant_state_passing_fwd(
                                rearrange(states, "... p n -> ... (p n)"),
                                dA_cumsum[:, :, :, -1],
                                d_state_val,
                                initial_states=rearrange(initial_states, "... p n -> ... (p n)") \
                                    if initial_states is not None else None,
                                seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=mm_dtype
                            )

    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]
    # if getattr(_quant_mamba_chunk_scan_combined_fwd, "_dump_input_this_call", False):
    #     torch.set_printoptions(threshold=float('inf'))
    #     print("rearrange_states=",states.shape)
    #     first_head_data = states[0, :, 0, :,:] 
    #     print("states_shape:", first_head_data.shape)
    #     print("states=",first_head_data)
        # print(f"states=",states)
        # print(f"states_shape",states.shape)

    CB = _quamba2_bmm_chunk_fwd(q_C, C_scale, q_B, B_scale, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    # if getattr(_quant_mamba_chunk_scan_combined_fwd, "_dump_input_this_call", False):
    #     torch.set_printoptions(threshold=float('inf'))
    #     print(f"q_c={q_C}")
    #     print(f"q_c_shape={q_C.shape}")
    #     print(f"q_C_scale={C_scale}")
    #     print(f"q_b={q_B}")
    #     print(f"q_b_shape={q_B.shape}")
    #     print(f"q_b_scale={B_scale}")
    #     print("CB_shape=",CB.shape)
    #     print("CB=",CB)
    
    # B, L, G, D = q_C.shape 
    # pattern_seq = torch.arange(L, device=q_C.device, dtype=torch.int8) % 8
    # pattern_reshaped = pattern_seq.view(1, L, 1, 1)
    # q_C = pattern_reshaped.expand(B, L, G, D).contiguous()
    # C_scale = torch.full_like(C_scale, 0.25)

    # if getattr(_quant_mamba_chunk_scan_combined_fwd, "_dump_input_this_call", False):
    #     _dump_bin("_quamba2_chunk_scan_fwd_q_C", q_C)
    #     _dump_bin("_quamba2_chunk_scan_fwd_C_scale", C_scale)
    out, out_x = _quamba2_chunk_scan_fwd(
        CB, q_x, x_scales, x_head_group_range, x_dim_group_range, dt, dA_cumsum, q_C, C_scale, states,
        q_D=q_D, D_scale=D_scale, q_z=q_z, z_scale=z_scale,
        seq_idx=seq_idx, mm_dtype=torch.float16
    )

    # if getattr(_quant_mamba_chunk_scan_combined_fwd, "_dump_input_this_call", False):
    #     torch.set_printoptions(threshold=float('inf'))
        # print(f"q_c={q_C}")
        # print(f"q_c_shape={q_C.shape}")
        # print(f"q_C_scale={C_scale}")
        # print(f"out=",out[0,:,0,:])
        # print(f"out_shape",out.shape)
        # print(f"dA_cumsum_shape=",dA_cumsum.shape)
        # print(f"dA_cumsum=",dA_cumsum[0,0,:,:])

    final_states = _quamba2_quant_ssm_states(final_states, x_head_group_range, x_dim_group_range, ssm_state_scale)

    if cu_seqlens is None:
        return out, final_states
    else:
        raise NotImplementedError("Only supports `cu_seqlens=None`")
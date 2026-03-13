
## 1. 本次真正的更新量（相对 V1）

## 1.1 FXP：不作为版本差异

- `fxp_utils.py` / `quamba/fxp_units.py` 这一套 FXP 非线性（exp/softplus/silu/ln）在 V1 已存在。
- V2 的主线目标不是改 FXP 算法本体，而是改 POT 的施加位置与粒度。

> 结论：**V1→V2 的核心差异在 POT，不在 FXP。**

---

## 1.2 POT：V1（老版） vs V2（新版）

### V1（老版 POT，粒度较粗）
1. 校准阶段的 activation scale 主要按 observer 原始结果使用（没有统一对关键 op 做 POT 圆整）。
2. `qChunkScan` 中 `A_log`、`D` 的量化走普通 absmax scale（非强制 POT scale）。

### V2（新版 POT，粒度更细）
1. 在 `run_quamba2_calibration` 中，对关键激活路径统一做 POT 圆整：
   - 新增 `POT_OP_PREFIXES`：
     `z_act, x_conv_out, B_conv_out, C_conv_out, dt_act, ssm_state_act, ssd_out_act`
   - 对这些 op 的 scale 执行 `round_scale_to_power_of_two(scale, mode="floor")`
2. 新增 `quantize_tensor_per_tensor_absmax_pot(...)`，支持直接产出 POT scale。
3. 将 `qChunkScan` 中 `A_log`、`D` 的量化替换为 POT 量化接口。

---

## 2. V2 中 POT 替换落点（代码级）

- `quamba/modelutils_mamba.py`
  - 新增：`POT_OP_PREFIXES`
  - 新增：`round_scale_to_power_of_two(...)`
  - `run_quamba2_calibration(...)` 内对命中前缀的 scale 做 `floor` 到 2^k

- `quamba/quant_utils.py`
  - 新增：`quantize_tensor_per_tensor_absmax_pot(...)`

- `quamba/qChunkScan.py`
  - `A_log` / `D`：
    - V1：`quantize_tensor_per_tensor_absmax(...)`
    - V2：`quantize_tensor_per_tensor_absmax_pot(...)`

> 这三处是本次“老POT→新POT”的主替换链路。

---

## 3. 评估怎么做（V1 vs V2）

## 3.1 脚本
- V1历史评估：`evaluate_quamba_fxp.sh`（旧日志在 `logs/fxp_*`、`logs/latest_*`）
- V2评估：`evaluate_quamba_newpot_fxp.sh`（日志在 `logs/newpot_*`）

## 3.2 公平对比要求（必须一致）
1. 同模型（如 `quamba2-2.7b-w8a8`）
2. 同任务（lambada_openai / winogrande）
3. 同 seed（`QUAMBA_SEED`）
4. 同 FXP 开关组合（none / exp / softplus / silu / all）
5. 同 slopefix 模式和 conv1d_silu 配置
6. 隔离 Triton cache（避免误复用）

---

## 4. 已有实验记录（按“旧POT+FXP”与“新POT+FXP”归档）

> 注意：部分 json 文件是“连续多次实验拼接”，同一文件里有多组配置。下面只摘录可定位结果，并标注来源。

## 4.1 旧版（V1：老POT+FXP）

### (A) Winogrande（固定 seed 历史记录）
来源：
- `logs/latest_none_after_ln1pexp.txt`
- `logs/latest_silu_slopefix_ln1pexp.txt`
- `logs/latest_exp+softplus+silu_after_ln1pexp.txt`

结果：
- none：`acc 0.6338 ± 0.0135`
- silu-only：`acc 0.6290 ± 0.0136`
- all-fxp：`acc 0.6440 ± 0.0135`

### (B) Lambada（旧版历史）
来源：`logs/fxp_none/quamba2-2.7b-w8a8_fp16.json`
- none：`acc≈0.6654, ppl≈4.4227`

来源：`logs/fxp_exp+softplus+silu/quamba2-2.7b-w8a8_fp16.json`（多次运行）
- all-fxp（历史高值之一）：`acc≈0.6996, ppl≈3.8602`

---

## 4.2 新版（V2：新POT+FXP）

来源：
- `logs/newpot_none/quamba2-2.7b-w8a8_fp16.json`
- `logs/newpot_exp+softplus+silu/quamba2-2.7b-w8a8_fp16.json`
- `logs/newpot_exp/*.json`, `logs/newpot_softplus/*.json`, `logs/newpot_silu/*.json`

可读到的代表记录：
- none：lambada `acc=0.6724, ppl=4.3177`；winogrande `acc=0.6338`
- all-fxp（同文件内多次运行，存在配置差异）有：
  - 稳定较好记录：`acc=0.7039, ppl=3.7675`
  - 也有异常记录（如 acc 很低、ppl 极高），说明该文件混入了不稳定配置/中间试验

---

## 5. 当前分析结论（纠正版）

1. **这次版本更新的主变量是 POT 粒度**，不是 FXP 是否引入。
2. 从已有日志看，新POT在若干配置上有潜在收益（如 lambada none / all-fxp 出现更优记录），但由于日志混入多配置，需要严格同参重跑做最终结论。
3. 复现实验时，conv1d_silu 与 slopefix 模式对结果影响非常大；比较 V1/V2 时必须锁死这些变量，否则会把“配置差异”误判为“POT差异”。

---

## 6. 给新对话助手的执行清单（最实用）

1. 先确认：本轮任务只比较 **老POT vs 新POT**，FXP 配置保持不变。
2. 先跑 3 组最小对照（同 seed）：
   - none
   - silu-only
   - all-fxp（exp,softplus,silu）
3. 每组固定：`QUAMBA_FXP_SILU_CONV1D`、`SLOPEFIX_MODE_*`、`LN_WIDTH`。
4. 统一任务（建议先 lambada_openai + winogrande）。
5. 输出成对表格：`V1指标 / V2指标 / 差值`。

---

## 7. 一句话总结

**V1 到 V2 不是“加了 FXP”，而是“在保持 FXP 不变的前提下，把 POT 从粗粒度升级到细粒度并落到关键路径（校准激活 + A_log/D）”。**

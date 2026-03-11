import datasets
import torch
import torch.nn.functional as F
import os
from transformers import AutoTokenizer
from tqdm import tqdm
from quamba.quamba_mixer_seq import QuambaLMHeadModel

# ---------------- config ----------------
model_path = "/deltadisk/congxiao/code/github/Quamba/pretrained_models/ut-enyac/quamba2-2.7b-w8a8" 
data_dir = "/deltadisk/congxiao/dataset/winogrande/winogrande_xl"

DEVICE = "cuda"
DEBUG = True
# ---------------------------------------

def doc_to_target(doc):
    """lm-eval 原版函数：获取 continuation"""
    idx = doc["sentence"].index("_") + 1
    target = doc["sentence"][idx:].strip()
    if target != ".":
        target = " " + target
    return target

def doc_to_choice(doc):
    """lm-eval 原版函数：获取两个 context"""
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]

def doc_to_label(doc):
    """获取正确答案的索引 (0 或 1)"""
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]

def get_loglikelihood(model, tokenizer, context, continuation):
    """
    计算 log P(continuation | context)
    完全按照 lm-eval 的方式
    """
    # 编码 context
    ctx_enc = tokenizer.encode(context, add_special_tokens=False)
    
    # 编码 context + continuation
    full_text = context + continuation
    full_enc = tokenizer.encode(full_text, add_special_tokens=False)
    
    # 找到 continuation 的起始位置
    # 由于 BPE，需要找到第一个不同的位置
    ctx_len = len(ctx_enc)
    cont_start = ctx_len
    
    for i in range(min(ctx_len, len(full_enc))):
        if i >= len(ctx_enc) or ctx_enc[i] != full_enc[i]:
            cont_start = i
            break
    
    cont_start = min(cont_start, len(full_enc) - 1)
    cont_start = max(cont_start, 0)
    
    num_cont_tokens = len(full_enc) - cont_start
    
    if num_cont_tokens <= 0:
        return float('-inf'), 0
    
    input_ids = torch.tensor([full_enc], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits if hasattr(out, "logits") else out
        logits = logits.float()
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        total_ll = 0.0
        actual_tokens = 0
        
        for i in range(cont_start, len(full_enc)):
            if i == 0:
                continue
            token_id = full_enc[i]
            token_ll = log_probs[0, i - 1, token_id].item()
            total_ll += token_ll
            actual_tokens += 1
    
    return total_ll, actual_tokens

def evaluate_sample(model, tokenizer, sample, debug=False):
    """
    按照 lm-eval 方式评估单个样本
    
    multiple_choice 评估：
    - choices = [context + option1, context + option2]
    - target = suffix (continuation)
    - 比较 P(target | choice1) vs P(target | choice2)
    """
    if "_" not in sample["sentence"]:
        return None, None, {}
    
    # 使用 lm-eval 的预处理函数
    choices = doc_to_choice(sample)  # [ctx+opt1, ctx+opt2]
    target = doc_to_target(sample)   # suffix (continuation)
    gold = doc_to_label(sample)      # 0 或 1
    
    # 计算每个 choice 的 log likelihood
    ll0, ntok0 = get_loglikelihood(model, tokenizer, choices[0], target)
    ll1, ntok1 = get_loglikelihood(model, tokenizer, choices[1], target)
    
    # acc: 直接比较（lm-eval 默认使用这个）
    pred = 0 if ll0 > ll1 else 1
    
    debug_info = {
        "sentence": sample["sentence"],
        "choice0": choices[0],
        "choice1": choices[1],
        "target": target,
        "ll0": ll0,
        "ll1": ll1,
        "ntok0": ntok0,
        "ntok1": ntok1,
        "pred": pred,
        "gold": gold
    }
    
    if debug:
        print(f"\nSentence: {sample['sentence']}")
        print(f"Choice 0: '{choices[0]}'")
        print(f"Choice 1: '{choices[1]}'")
        print(f"Target: '{target}'")
        print(f"LL0: {ll0:.4f} ({ntok0} tokens)")
        print(f"LL1: {ll1:.4f} ({ntok1} tokens)")
        print(f"Pred: {pred}, Gold: {gold}, Correct: {pred == gold}")
    
    return pred, gold, debug_info

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    print(f"Loading model from {model_path}...")
    model = QuambaLMHeadModel.from_pretrained(model_path, device="cuda")
    model.eval()
    print(model)
    
    # 加载数据
    print(f"Loading data from {data_dir}...")
    val_files = [f for f in os.listdir(data_dir) if "validation" in f and f.endswith(".parquet")]
    
    if val_files:
        full_val_paths = [os.path.join(data_dir, f) for f in val_files]
        ds = datasets.load_dataset("parquet", data_files={"validation": full_val_paths}, split="validation")
    else:
        ds = datasets.load_dataset("winogrande", "winogrande_xl", split="validation")
    
    print(f"Total samples: {len(ds)}")
        
    # 正式评估
    correct = 0
    total = 0
    
    print("\nEvaluating WinoGrande (lm-eval style)...")
    
    for sample in tqdm(ds):
        pred, gold, _ = evaluate_sample(model, tokenizer, sample, debug=False)
        if pred is not None:
            if pred == gold:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"Model: {model_path}")
    print(f"Dataset: winogrande_xl validation ({total} samples)")
    print("=" * 60)
    print(f"acc: {accuracy:.4%} ({correct}/{total})")
    print("=" * 60)
    
    print("\nThis should match lm-evaluation-harness results.")

if __name__ == "__main__":
    main()
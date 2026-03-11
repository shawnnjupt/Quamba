import datasets
import torch
import torch.nn.functional as F
import os
from transformers import GPTNeoXTokenizerFast
from tqdm import tqdm
from torch.utils.data import DataLoader
from quamba.quamba_mixer_seq import QuambaLMHeadModel

# ---------------- config ----------------
model_path = "/deltadisk/congxiao/code/github/Quamba/pretrained_models/ut-enyac/quamba2-370m-w8a8" 
tasks = {
    "ARC-Easy": "/deltadisk/congxiao/dataset/ai2_arc_c_e/ARC-Easy",
    "ARC-Challenge": "/deltadisk/congxiao/dataset/ai2_arc_c_e/ARC-Challenge"
}
EVAL_SPLIT = "test"
DEVICE = "cuda"
BATCH_SIZE = 8
# ---------------------------------------

def arc_collate_fn(batch, tokenizer):
    all_input_ids = []
    all_loss_mask = []
    gold_labels = []
    choices_per_sample = []
    
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '1': 0, '2': 1, '3': 2, '4': 3, '5': 4}

    for sample in batch:
        question = sample["question"].strip()
        choices_raw = sample["choices"]
        answer_key = str(sample["answerKey"])
        
        num_choices = len(choices_raw["text"])
        choices_per_sample.append(num_choices)
        
        # 标准 Prompt
        context_text = f"Question: {question}\nAnswer:"
        context_enc = tokenizer(context_text, add_special_tokens=False)
        context_len = len(context_enc.input_ids)

        for i in range(num_choices):
            choice_text = choices_raw["text"][i]
            # 拼接时确保 Answer: 后有一个空格
            full_text = context_text + " " + choice_text
            full_enc = tokenizer(full_text, add_special_tokens=False)
            full_ids = full_enc.input_ids
            
            loss_mask = [0] * len(full_ids)
            for j in range(context_len, len(full_ids)):
                loss_mask[j] = 1
                
            all_input_ids.append(torch.tensor(full_ids))
            all_loss_mask.append(torch.tensor(loss_mask))

        gold_labels.append(label_map.get(answer_key, -1))

    input_ids = torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=tokenizer.eos_token_id)
    loss_mask = torch.nn.utils.rnn.pad_sequence(all_loss_mask, batch_first=True, padding_value=0)
    attention_mask = (input_ids != tokenizer.eos_token_id).long()

    return input_ids, attention_mask, loss_mask, torch.tensor(gold_labels), choices_per_sample

def compute_scores(logits, input_ids, loss_mask):
    """
    同时返回原始 Log-Likelihood 和 归一化后的 Log-Likelihood
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_loss_mask = loss_mask[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)
    per_token_logps = per_token_logps * shift_loss_mask
    
    sum_logps = per_token_logps.sum(dim=1)
    num_tokens = shift_loss_mask.sum(dim=1)
    
    norm_logps = sum_logps / (num_tokens + 1e-8)
    
    return sum_logps, norm_logps

def evaluate_single_task(model, tokenizer, task_name, task_dir):
    print(f"\n>>> Evaluating {task_name}...")
    files = [f for f in os.listdir(task_dir) if EVAL_SPLIT in f and f.endswith(".parquet")]
    if not files: return None
    
    full_paths = [os.path.join(task_dir, f) for f in files]
    ds = datasets.load_dataset("parquet", data_files={EVAL_SPLIT: full_paths}, split=EVAL_SPLIT)
    dataloader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=lambda x: arc_collate_fn(x, tokenizer))

    correct_raw = 0
    correct_norm = 0
    total = 0

    for batch in tqdm(dataloader, desc=f"Running {task_name}"):
        input_ids, attention_mask, loss_mask, gold_labels, choices_per_sample = batch
        input_ids, attention_mask, loss_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE), loss_mask.to(DEVICE)

        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits if hasattr(out, "logits") else out
            
            raw_scores, norm_scores = compute_scores(logits, input_ids, loss_mask)
            
            current_idx = 0
            for i, num_choices in enumerate(choices_per_sample):
                # 原始得分预测
                s_raw = raw_scores[current_idx : current_idx + num_choices]
                if torch.argmax(s_raw).item() == gold_labels[i].item():
                    correct_raw += 1
                
                # 归一化得分预测
                s_norm = norm_scores[current_idx : current_idx + num_choices]
                if torch.argmax(s_norm).item() == gold_labels[i].item():
                    correct_norm += 1
                
                total += 1
                current_idx += num_choices

    # 根据任务类型选择返回哪种准确率
    # ARC-Easy 通常看 Raw，ARC-Challenge 通常看 Norm
    acc_raw = correct_raw / total
    acc_norm = correct_norm / total
    
    return acc_raw, acc_norm, total

def main():
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_path}...")
    model = QuambaLMHeadModel.from_pretrained(model_path, device="cuda")
    model.eval()
    print(model)
    
    results = {}
    for task_name, task_dir in tasks.items():
        acc_raw, acc_norm, count = evaluate_single_task(model, tokenizer, task_name, task_dir)
        results[task_name] = (acc_raw, acc_norm, count)

    print("\n" + "="*75)
    print(f"{'Task Name':<20} | {'Acc (Raw)':<12} | {'Acc (Norm)':<12} | {'Total'}")
    print("-" * 75)
    for name, (acc_raw, acc_norm, total) in results.items():
        # 标注一下哪个是对齐论文的
        star_raw = "*" if "Easy" in name else " "
        star_norm = "*" if "Challenge" in name else " "
        print(f"{name:<20} | {acc_raw:<12.4%}{star_raw} | {acc_norm:<12.4%}{star_norm} | {total}")
    print("="*75)
    print("注：带 * 的分数通常是论文中报告的标准指标。")

if __name__ == "__main__":
    main()
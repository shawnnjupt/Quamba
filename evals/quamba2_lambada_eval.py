import datasets
import torch
import numpy as np
from transformers import GPTNeoXTokenizerFast
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from tqdm import tqdm
from torch.utils.data import DataLoader
from quamba.quamba_mixer_seq import QuambaLMHeadModel


# ---------------- config ----------------
model_path = "/deltadisk/congxiao/code/github/Quamba/pretrained_models/ut-enyac/quamba2-130m-w8a8" 
parquet_path = "/deltadisk/congxiao/dataset/lambada_openai/en/test/en.parquet"

DEVICE = "cuda"
DTYPE = torch.float16
BATCH_SIZE = 64  # 你可以根据显存调整，如 8, 16, 32
# ---------------------------------------

def collate_fn(batch, tokenizer):
    """
    自定义 Collate 函数：处理不同长度的文本并生成 target_mask
    """
    texts = []
    target_masks = []
    full_ids_list = []
    
    for sample in batch:
        text = sample["text"].strip()
        pos = text.rfind(" ")
        if pos == -1: continue
        
        context = text[:pos]
        
        # 编码
        full_enc = tokenizer.encode(text, add_special_tokens=False)
        ctx_enc = tokenizer.encode(context, add_special_tokens=False)
        
        tgt_len = len(full_enc) - len(ctx_enc)
        if tgt_len <= 0: continue
        
        # 构造该样本的 mask: context 部分为 0, target 部分为 1
        mask = [0] * len(ctx_enc) + [1] * tgt_len
        
        full_ids_list.append(torch.tensor(full_enc))
        target_masks.append(torch.tensor(mask))

    if not full_ids_list:
        return None

    # Padding: 使用 tokenizer 的 pad_token (通常是 eos_token)
    # input_ids 形状: [batch, max_len]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        full_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    # target_mask 形状: [batch, max_len]
    # 注意：pad 的地方 mask 也是 0
    target_mask = torch.nn.utils.rnn.pad_sequence(
        target_masks, batch_first=True, padding_value=0
    )
    
    return input_ids, target_mask

def main():
    # 1. 加载 Tokenizer
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right' # 评测 NLL 建议用 right padding
    
    print(f"Loading model from {model_path}...")
    model = QuambaLMHeadModel.from_pretrained(model_path, device="cuda")
    model.eval()
    print(model)
    # 2. 加载数据
    ds = datasets.load_dataset("parquet", data_files={"test": parquet_path}, split="test")
    
    # 使用 DataLoader 进行多 Batch 处理
    dataloader = DataLoader(
        ds, 
        batch_size=BATCH_SIZE, 
        collate_fn=lambda x: collate_fn(x, tokenizer)
    )

    total_instances = 0
    correct_instances = 0
    total_nll_sum = 0.0

    # 使用 CrossEntropyLoss，设置 reduction='none' 以便手动处理 mask
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    print(f"Starting evaluation (Batch Size: {BATCH_SIZE})...")
    for batch in tqdm(dataloader):
        if batch is None: continue
        
        input_ids, target_mask = batch
        input_ids = input_ids.to(DEVICE)
        target_mask = target_mask.to(DEVICE)

        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits if hasattr(out, "logits") else out

        # Shift 逻辑
        # logits[:, i, :] 预测的是 input_ids[:, i+1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        # target_mask 也要 shift，因为我们要对齐预测位置
        shift_mask = target_mask[:, 1:].contiguous()

        # 计算所有位置的 loss [batch, seq_len-1]
        losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        ).view(shift_labels.size())

        # 只保留 target 部分的 loss
        target_losses = losses * shift_mask
        
        # 遍历 Batch 计算每个样本的指标
        for i in range(input_ids.size(0)):
            sample_mask = shift_mask[i]
            if not sample_mask.any(): continue
            
            # 1. 该样本的总 NLL (用于 PPL)
            sample_nll = target_losses[i].sum().item()
            total_nll_sum += sample_nll
            
            # 2. 该样本的 Accuracy (全词匹配)
            # 提取该样本 target 对应的预测和标签
            sample_preds = shift_logits[i][sample_mask.bool()].argmax(dim=-1)
            sample_targets = shift_labels[i][sample_mask.bool()]
            
            if torch.equal(sample_preds, sample_targets):
                correct_instances += 1
            
            total_instances += 1

    # 3. 计算最终指标
    if total_instances == 0:
        print("No valid samples.")
        return

    avg_acc = correct_instances / total_instances
    # 对齐论文 5.02 的 Per-Word PPL 公式
    final_ppl = np.exp(total_nll_sum / total_instances)

    print("\n" + "="*50)
    print(f"Model: {model_path}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Total Samples: {total_instances}")
    print(f"Accuracy: {avg_acc:.4f}")
    print(f"Perplexity (Per-Word): {final_ppl:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
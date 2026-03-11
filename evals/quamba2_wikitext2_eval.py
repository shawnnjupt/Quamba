''' 
Description: Mamba/Mamba2 Evaluation on WikiText-2 (Perplexity)
Based on Llama2 evaluation logic provided by user.
''' 

import datasets 
import logging 
import sys 
import torch 
import numpy as np 
from tqdm import tqdm 
from transformers import AutoTokenizer 

# Mamba specific imports
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from quamba.quamba_mixer_seq import QuambaLMHeadModel


logger = logging.getLogger(__name__) 

logging.basicConfig( 
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", 
    datefmt="%m/%d/%Y %H:%M:%S", 
    handlers=[logging.StreamHandler(sys.stdout)], 
) 

# ========================== 配置区域 ==========================
# Configuration 
task_name = "wikitext-2-raw-v1" 
dataset_path = '/deltadisk/congxiao/dataset/wikitext-2-raw-v1' 
model_path = "/deltadisk/congxiao/code/github/Quamba/pretrained_models/ut-enyac/quamba2-370m-w8a8" 

# 评估参数
SEQ_LEN = 2048  # Mamba 上下文长度通常较长，但为了对比通常设为 2048
EVAL_BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 # Mamba 通常使用 fp16 或 bf16
# ================================================================

def evaluate_mamba_perplexity(model_path, dataset_path, seqlen=2048, batch_size=1): 
    """ 
    使用 Mamba 模型在 WikiText-2 上计算 Perplexity (PPL)
    """ 
    print(f"Starting Mamba evaluation with batch_size = {batch_size}...") 
    
    try: 
        # 1. 加载 Tokenizer
        # Mamba 官方权重通常使用 gpt-neox-20b 的 tokenizer，如果是 Mamba2 或微调版，请指向具体路径
        print(f"Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            print("Warning: Could not load tokenizer from model path, falling back to 'EleutherAI/gpt-neox-20b'")
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            
        if tokenizer.pad_token is None: 
            tokenizer.pad_token = tokenizer.eos_token 

        # 2. 加载 Mamba 模型
        print(f"Loading Mamba model from {model_path}...")
        model = QuambaLMHeadModel.from_pretrained(model_path, device="cuda")
        model.eval()
        print(model)
        # print("d_state =", model.backbone.layers[0].mixer.d_state)
        # print("chunk_size =", model.backbone.layers[0].mixer.chunk_size)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    except Exception as e: 
        print(f"Error loading model: {e}") 
        return None 
    
    # 3. 加载数据集
    print(f"Loading dataset from {dataset_path}...")
    try:
        # 尝试加载本地数据集
        test_dataset = datasets.load_dataset(dataset_path, split="test") 
    except:
        # 如果本地失败，尝试在线加载
        print("Local dataset not found, trying to download...")
        test_dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # 4. 数据预处理 (与 Llama2 代码逻辑一致)
    text = "\n\n".join(test_dataset["text"]) 
    print(f"Total text length: {len(text)} characters") 
    
    print("Tokenizing entire text...") 
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False) 
    input_ids = encodings.input_ids 
    
    print(f"Total sequence length: {input_ids.size(1)} tokens") 
    
    # 截断并重塑数据
    nsamples = input_ids.numel() // seqlen 
    input_ids = input_ids[:, :nsamples * seqlen].view(nsamples, seqlen) 
    
    print(f"Number of samples: {nsamples}") 
    print(f"Sequence length per sample: {seqlen}") 
    
    # 分批
    input_batches = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)] 
    nbatches = len(input_batches) 
    print(f"Number of batches: {nbatches}") 

    # 5. 计算 Perplexity
    nlls = [] 
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none") 
    
    print("Computing perplexity...") 
    for i in tqdm(range(nbatches), desc="Processing batches"): 
        batch_input_ids = input_batches[i].to(DEVICE) 
        
        with torch.no_grad(): 
            # Mamba Forward Pass
            # MambaLMHeadModel 的 forward 返回通常包含 logits
            outputs = model(batch_input_ids) 
            
            # 检查输出类型，获取 logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                # 某些版本的 mamba 实现直接返回 logits
                logits = outputs
            
            # 计算 Loss (Shift logits)
            shift_logits = logits[:, :-1, :].contiguous() 
            shift_labels = batch_input_ids[:, 1:].contiguous() 
            
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) 
            
            # 这里我们需要每个样本的 loss，上面的 view(-1) 混合了 batch。
            # 为了保持和你 Llama2 代码完全一致的逻辑（按样本求平均）：
            loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels) # (batch, seq_len-1)
            
            neg_log_likelihood = loss.float().mean(dim=1) 
            nlls.append(neg_log_likelihood) 
    
    nlls_tensor = torch.cat(nlls) 
    ppl = torch.exp(nlls_tensor.mean()) 
    
    print(f"\n" + "="*50) 
    print(f"MAMBA EVALUATION RESULTS:") 
    print(f"="*50) 
    print(f"Model: {model_path}")
    print(f"Total samples: {nsamples}") 
    print(f"Sequence length: {seqlen}") 
    print(f"Batch size: {batch_size}") 
    print(f"Average negative log-likelihood: {nlls_tensor.mean().item():.6f}") 
    print(f"Perplexity: {ppl.item():.4f}") 
    print(f"="*50) 
    
    return ppl.item() 

if __name__ == "__main__": 
    evaluate_mamba_perplexity(
        model_path=model_path, 
        dataset_path=dataset_path, 
        seqlen=SEQ_LEN, 
        batch_size=EVAL_BATCH_SIZE
    )
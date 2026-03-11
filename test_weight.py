import torch

path = "/deltadisk/congxiao/code/github/Quamba/pretrained_models/ut-enyac/quamba2-130m-w8a8/pytorch_model.bin"
state = torch.load(path, map_location="cpu")

print("===== MODEL STATE DICT SUMMARY =====")
print(f"Total keys: {len(state)}\n")

for k, v in state.items():
    print(f"\nKey: {k}")

    # 如果是 tensor
    if torch.is_tensor(v):
        print(f"  type: tensor")
        print(f"  dtype: {v.dtype}")
        print(f"  tensor——shape: {tuple(v.shape)}")

        # 对于标量张量，打印值
        if v.numel() <= 10000:
            print(f"  value: {v}")

    # 如果是数字
    elif isinstance(v, (float, int)):
        print(f"  type: {type(v).__name__}")
        print(f"  value: {v}")

    # 如果是列表
    elif isinstance(v, list):
        print(f"  type: list (len={len(v)})")

    # 其他类型（字典、自定义结构等）
    else:
        print(f"  type: {type(v)}")
        print(f"  value (str): {str(v)[:200]}...")
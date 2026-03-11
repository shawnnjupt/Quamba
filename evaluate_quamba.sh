CUDA_VISIBLE_DEVICES=1 python main.py quamba2-2.7b-w8a8 \
  --pretrained_dir /deltadisk/congxiao/code/github/Trash/Quamba/pretrained_models/ut-enyac \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --log_dir ./logs \
  --group_heads
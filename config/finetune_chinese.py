import time

out_dir = 'out-chinese-finetune'
eval_interval = 50   # 每 50 步評估一次，密切觀察轉變
eval_iters = 20      # 快速評估
wandb_log = False # feel free to turn on
wandb_project = 'chinese-finetune'
wandb_run_name = 'gentle-ft-' + str(time.time())

dataset = 'chinese_char'  # 使用 chinese_char 數據
init_from = 'resume'  # 從 shakespeare 模型開始 finetune

# 輕度 finetune 設定 - 觀察英文到中文的轉變過程
always_save_checkpoint = True

# 溫和的訓練設定
batch_size = 2  # 小 batch size，更細緻的學習
gradient_accumulation_steps = 4  # 小梯度累積
max_iters = 10000  # 適中的訓練步數，觀察漸進變化

block_size = 64  # 適中的 context length

# 溫和的學習率設定
learning_rate = 5e-6  # 極小學習率，緩慢學習
decay_lr = True  # 啟用學習率衰減
warmup_iters = 50  # 溫和 warmup
lr_decay_iters = 1000  # 學習率衰減步數
min_lr = 1e-6  # 最小學習率

# 保持一些原有特徵
dropout = 0.02  # 很小的 dropout，保留更多原始知識

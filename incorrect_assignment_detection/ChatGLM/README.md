# LLM-based-IND

## 简介
- 微调ChatGLM3-6b 32k
- 环境需求
    - transformers
    - deepspeed 
    - accelerate
    - peft
- 可以在训练前配置wandb监控实验进程

## 训练
bash script/ds_lora_epoch.sh

## 推理
python inference.py
# LLM-based-IND

## intro
- finetune ChatGLM3-6b 32k
- environment requirements
    - transformers
    - deepspeed 
    - accelerate
    - peft
    - scikit-lean
    - imbalanced-learn

## train
bash train.sh

## valid&test
multi GPU
- accelerate launch --num_processes 8 inference.py --lora_path your_lora_checkpoint --model_path path_to_chatglm --pub_path path_to_pub_file --eval_path path_to_eval_author --saved_dir result_to_save

single card
- accelerate launch --num_processes 1 --gpu_ids='0' inference.py --lora_path your_lora_checkpoint --model_path path_to_chatglm --pub_path path_to_pub_file --eval_path path_to_eval_author --saved_dir result_to_save

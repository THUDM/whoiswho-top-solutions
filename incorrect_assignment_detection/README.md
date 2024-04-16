# Incorrect Assignment Detection

## Prerequisites
- Linux
- Python 3.10
- PyTorch 2.2.0+cu121
  
## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/THUDM/whoiswho-top-solutions.git
cd whoiswho-top-solutions/incorrect_assignment_detection
```

For ``GCN``, 
```bash
pip install -r GCN/requirements.txt
```

For ``GCCAD``,
```bash
pip install -r GCCAD/requirements.txt
```

For ``ChatGLM``,
```bash
pip install -r ChatGLM/requirements.txt
```

## IND Dataset
The dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1_CX50fRxou4riEHzn5UYKg?pwd=gvza) with password gvza, [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/IND-WhoIsWho/IND-WhoIsWho.zip) or [DropBox](https://www.dropbox.com/scl/fi/o8du146aafl3vrb87tm45/IND-WhoIsWho.zip?rlkey=cg6tbubqo532hb1ljaz70tlxe&dl=1).
Unzip the dataset and put files into ``dataset/`` directory.

## Run Baselines for [KDD Cup 2024](https://www.biendata.xyz/competition/ind_kdd_2024/)

We provide three baselines: [GCN](https://arxiv.org/abs/1609.02907), [GCCAD](https://arxiv.org/abs/2108.07516), and [ChatGLM](https://arxiv.org/abs/2210.02414) [[Hugging Face]](https://huggingface.co/THUDM/chatglm3-6b-32k). A fine-tuned ChatGLM checkpoint via Lora can be downloaded from [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/IND-WhoIsWho/ind_chatglm_ckpt_1000.zip).

```bash
export CUDA_VISIBLE_DEVICES='?'  # specify which GPU(s) to be used

# Method 1 & 2: GCN & GCCAD
cd GCCAD  # or ``cd GCN``
python encoding.py --path ../dataset/pid_to_info_all.json --save_path ../dataset/roberta_embeddings.pkl
python build_graph.py --author_dir ../dataset/train_author.json  --save_dir ../dataset/train.pkl
python build_graph.py
python train.py  --train_dir ../dataset/train.pkl  --test_dir ../dataset/valid.pkl


# Method 3: ChatGLM (Test Environment: 8 * A100)
cd ChatGLM
bash train.sh
accelerate launch --num_processes 8 inference.py --lora_path your_lora_path --model_path your_model_path --pub_path  ../dataset/pid_to_info_all.json --eval_path ../dataset/ind_valid_author.json  # multi-GPU
python inference.py --lora_path your_lora_checkpoint --model_path path_to_chatglm --pub_path ../dataset/pid_to_info_all.json  --eval_path ../dataset/ind_valid_author.json   # single GPU
```

## Results on Valiation Set

|  Method  | AUC   |
|-------|-------|
| GCN  | 0.58625 |
| GCCAD | 0.63451 |
| ChatGLM  | 0.71385 |

## Citation

If you find this repo useful in your research, please cite the following papers:

```
@inproceedings{chen2023web,
  title={Web-scale academic name disambiguation: the WhoIsWho benchmark, leaderboard, and toolkit},
  author={Bo Chen and Jing Zhang and Fanjin Zhang and Tianyi Han and Yuqing Cheng and Xiaoyan Li and Yuxiao Dong and Jie Tang},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={3817--3828},
  year={2023}
}

@article{zhang2024oag,
    title={OAG-Bench: A Human-Curated Benchmark for Academic Graph Mining},
    author={Fanjin Zhang and Shijie Shi and Yifan Zhu and Bo Chen and Yukuo Cen and Jifan Yu and Yelin Chen and Lulu Wang and Qingfei Zhao and Yuqing Cheng and Tianyi Han and Yuwei An and Dan Zhang and Weng Lam Tam and Kun Cao and Yunhe Pang and Xinyu Guan and Huihui Yuan and Jian Song and Xiaoyan Li and Yuxiao Dong and Jie Tang},
    journal={arXiv preprint arXiv:2402.15810},
    year={2024}
}
```

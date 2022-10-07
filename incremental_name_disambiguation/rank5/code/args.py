# -*- coding: utf-8 -*-
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    # 数据存储位置
    parser.add_argument("--datapath", type=str, default="./data",help="解压后的数据存储位置")
    parser.add_argument("--featpath", type=str, default="./work/feature",help="构造特征存储位置")
    parser.add_argument("--subpath",  type=str, default="./work/submit",help="提交结果存储位置")
    parser.add_argument("--w2vpath",  type=str, default="./work/w2v",help="w2v模型存储位置")
    parser.add_argument("--outpath",  type=str, default="./work/output",help="其他输出文件存储位置")
    parser.add_argument("--modelpath",  type=str, default="./work/model")
    
    parser.add_argument("--model", type=str, default="LightGBM",help='训练模型')
    parser.add_argument("--stratified", type=bool, default=True)
    parser.add_argument("--mode", type=str, default="online")
    parser.add_argument("--seed", type=int, default=2021)
    
    args = parser.parse_args()
    
    
    args.featpath = os.path.join(args.featpath,args.mode)
    args.subpath  = os.path.join(args.subpath,args.mode)
    args.outpath  = os.path.join(args.outpath,args.mode)
    args.w2vpath  = os.path.join(args.w2vpath,args.mode)
    
    # 新建文件夹
    os.makedirs(args.featpath,exist_ok =True)
    os.makedirs(args.subpath ,exist_ok =True)
    os.makedirs(args.outpath ,exist_ok =True)
    os.makedirs(args.w2vpath ,exist_ok =True)
    
    return args
    
if __name__=="__main__":
    
    args = get_parser()


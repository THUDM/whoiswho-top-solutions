import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import random
import json
import pickle
import logging
from torch.optim.lr_scheduler import _LRScheduler
from models import  GraphCAD, outlierLoss
from utils import *
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# python train.py --train_dir ../GCCAD/train.pkl --test_dir ../GCCAD/valid.pkl
def add_arguments(args):
    # essential paras
    args.add_argument('--train_dir', type=str, help="train_dir", default = "train.pkl")
    args.add_argument('--eval_dir', type=str, help="eval_dir", default = None)
    args.add_argument('--test_dir', type=str, help="test_dir", default = None)
    args.add_argument('--saved_dir', type=str, help="model save dir", default= "saved_model")
    args.add_argument('--log_name', type=str, help="log_name", default = "log")

    # training paras.
    args.add_argument('--epochs', type=int, help="training #epochs", default=50)
    args.add_argument('--seed', type=int, help="seed", default=1)
    args.add_argument('--lr', type=float, help="learning rate", default=5e-4)
    args.add_argument('--min_lr', type=float, help="min lr", default=2e-4)
    args.add_argument('--input_dim', type=int, help="input dimension", default=768)
    args.add_argument('--out_dim', type=int, help="output dimension", default=768)
    args.add_argument('--verbose', type=int, help="eval", default=1)

    # model paras.
    args.add_argument('--outer_layer', type=int, help="#layers of GraphCAD", default = 2)
    args.add_argument('--inner_layer', type=int, help="#layers of node_update", default = 1)
    args.add_argument('--is_global', help="whether to add global information", action = "store_false")
    args.add_argument('--is_edge', help="whether to use edge update", action = "store_false")
    args.add_argument('--pooling', type=str, help="pooing_type", choices=['memory', 'avg', 'min', 'max'], default = "memory")

    args.add_argument('--is_lp', help="whether to use link prediction loss", action = "store_false")
    args.add_argument("--lp_weight", type = float, help="the weight of link prediction loss", default=0.1)

    # dataset graph paras
    args.add_argument('--usecoo', help="use co-organization edge", action='store_true')
    args.add_argument('--usecov', help="use co-venue edge", action='store_true')
    args.add_argument('--threshold', type=float, help="threshold of coo and cov", default=0)
    args = args.parse_args()
    return args

def logging_builder(args):
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(os.path.join(os.getcwd(), args.log_name), mode='w')
    fileHandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger

class WarmupLinearLR(_LRScheduler):
    def __init__(self, optimizer, step_size, min_lr, peak_percentage=0.1, last_epoch=-1):
        self.step_size = step_size
        self.peak_step = peak_percentage * step_size
        self.min_lr = min_lr
        super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ret = []
        for tmp_min_lr, tmp_base_lr in zip(self.min_lr, self.base_lrs):
            if self._step_count <= self.peak_step:
                ret.append(tmp_min_lr + (tmp_base_lr - tmp_min_lr) * self._step_count / self.peak_step)
            else:
                ret.append(tmp_min_lr + max(0, (tmp_base_lr - tmp_min_lr) * (self.step_size - self._step_count) / (self.step_size - self.peak_step)))
        return ret

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args = add_arguments(args)
    setup_seed(args.seed)
    logger = logging_builder(args)
    logger.info(args)
    print(args)
    os.makedirs(os.path.join(os.getcwd(), args.saved_dir), exist_ok = True)

    encoder = GraphCAD(logger, args, args.input_dim, args.out_dim, args.outer_layer, args.inner_layer, is_global = args.is_global, is_edge = args.is_edge, pooling= args.pooling).cuda()
    criterion = outlierLoss(args, logger, is_lp = args.is_lp, lp_weight = args.lp_weight).cuda()
    
    with open(args.train_dir, 'rb') as files:
        train_data = pickle.load(files)

    if args.eval_dir is not None:
        with open(args.eval_dir, 'rb') as files:
            eval_data = pickle.load(files)
    else: #split train and valid
        random.shuffle(train_data)
        eval_data = train_data[int(len(train_data)*0.7):]
        train_data = train_data[:int(len(train_data)*0.7)]

    logger.info("# Batch: {} - {}".format(len(train_data), len(train_data) ))
    optimizer = torch.optim.Adam([{'params': encoder.parameters(), 'lr': args.lr}])
    optimizer.zero_grad()

    max_step = int(len(train_data) * 10)
    logger.info("max_step: %d, %d,  %d"%(max_step, len(train_data), args.epochs))
    scheduler = WarmupLinearLR(optimizer, max_step, min_lr=[args.min_lr])

    encoder.train()
    epoch_num = 0
    max_map = -1
    max_auc = -1
    max_epoch = -1
    early_stop_counter = 0
    for epoch_num in range(args.epochs):
        batch_loss = []
        batch_contras_loss = []
        batch_lp_loss = []
        batch_edge_score = []
        random.shuffle(train_data)
        for tmp_train in tqdm(train_data):

            batch_data, edge_labels,_,_ = tmp_train
            batch_data = batch_data.cuda()
            edge_labels = edge_labels.cuda()
            node_outputs, adj_matrix, adj_weight, labels, batch_item = batch_data.x, batch_data.edge_index, batch_data.edge_attr.squeeze(-1), batch_data.y, batch_data.batch

            if args.threshold > 0:
                flag = adj_weight[:,1:]<args.threshold
                adj_weight[:,1:] = torch.where(flag,torch.tensor(0.0),adj_weight[:,1:])
            if args.usecoo and args.usecov:
                adj_weight = adj_weight.mean(dim = -1)
            elif args.usecoo:
                adj_weight = (adj_weight[:,0] + adj_weight[:,1])/2
            elif args.usecov:
                adj_weight = (adj_weight[:,0] + adj_weight[:,2])/2
            else:
                adj_weight = adj_weight[:,0]
            flag = torch.nonzero(adj_weight).squeeze(-1)
            adj_matrix = adj_matrix.T[flag].T
            adj_weight = adj_weight[flag]
            edge_labels = edge_labels[flag]

            node_outputs, adj_weight, centroid, output_loss, centroid_loss, edge_prob = encoder(node_outputs, adj_matrix, adj_weight, batch_item, 1)
            overall_loss, _, contras_loss, lp_loss = criterion(output_loss, centroid_loss, edge_prob, edge_labels, adj_matrix, batch_item, labels, node_outputs, centroid)
            
            if lp_loss.isnan(): # edge labels are all 1
                overall_loss = contras_loss 
            
            batch_loss.append(overall_loss.item())
            batch_contras_loss.append(contras_loss.item())
            batch_lp_loss.append(lp_loss.item() if not lp_loss.isnan() else 0)
            optimizer.zero_grad()
            overall_loss.backward()   
            optimizer.step()
            scheduler.step()

        avg_batch_loss = np.mean(np.array(batch_loss))
        avg_batch_contras_loss = np.mean(np.array(batch_contras_loss))
        avg_batch_lp_loss = np.mean(np.array(batch_lp_loss))
        logger.info("Epoch:{} Overall loss: {:.6f} Contrastive loss: {:.6f} LP_loss: {:.6f}".format(epoch_num, avg_batch_loss, avg_batch_contras_loss, avg_batch_lp_loss))        
        
        if (epoch_num + 1) % args.verbose == 0:
            encoder.eval()
            test_loss = []
            test_contras_loss = []
            test_lp_loss = []
            test_gt = []

            labels_list = []
            scores_list = []
            with torch.no_grad():
                for tmp_test in tqdm(eval_data):
                    each_sub,_,_,_ = tmp_test
                    each_sub = each_sub.cuda()
                    node_outputs, adj_matrix, adj_weight, labels, batch_item = each_sub.x, each_sub.edge_index, each_sub.edge_attr.squeeze(-1), each_sub.y, each_sub.batch

                    if args.threshold > 0:
                        flag = adj_weight[:,1:]<args.threshold
                        adj_weight[:,1:] = torch.where(flag,torch.tensor(0.0),adj_weight[:,1:])
                    if args.usecoo and args.usecov:
                        adj_weight = adj_weight.mean(dim = -1)
                    elif args.usecoo:
                        adj_weight = (adj_weight[:,0] + adj_weight[:,1])/2
                    elif args.usecov:
                        adj_weight = (adj_weight[:,0] + adj_weight[:,2])/2
                    else:
                        adj_weight = adj_weight[:,0]
                    flag = torch.nonzero(adj_weight).squeeze(-1)
                    adj_matrix = adj_matrix.T[flag].T
                    adj_weight = adj_weight[flag]            

                    node_outputs, adj_weight, centroid, output_loss, centroid_loss, edge_prob = encoder(node_outputs, adj_matrix, adj_weight, batch_item, 1)
                    centroid = centroid.squeeze(0)
                    scores = criterion.get_score(node_outputs, centroid)

                    scores = scores.detach().cpu().numpy()
                    scores_list.append(scores)
                    labels = labels.detach().cpu().numpy()
                    test_gt.append(labels)

            auc, maps = MAPs(test_gt, scores_list)
            logger.info("Epoch: {} Auc: {:.6f} Maps: {:.6f} Max-Auc: {:.6f} Max-Maps: {:.6f}".format(epoch_num, auc, maps, max_auc, max_map))
            
            if maps > max_map or auc > max_auc:
                early_stop_counter = 0

                max_epoch = epoch_num
                max_map = maps if maps > max_map else max_map
                max_auc = auc if auc > max_auc else max_auc
                torch.save(encoder, f"{args.saved_dir}/model_{str(epoch_num)}.pt")
                
                logger.info("***************** Epoch: {} Max Auc: {:.6f} Maps: {:.6f} *******************".format(epoch_num, max_auc, max_map))
            else:
                early_stop_counter += 1
                if early_stop_counter >= 10:
                    print("Early stop!")
                    break     
            encoder.train()
            optimizer.zero_grad()
    logger.info("***************** Max_Epoch: {} Max Auc: {:.6f} Maps: {:.6f}*******************".format(max_epoch, max_auc, max_map))

    logger.info("Loading best model...")
    
    encoder = torch.load(f"{args.saved_dir}/model_{str(max_epoch)}.pt")
    encoder.eval()

    with open(args.test_dir, 'rb') as f:
        test_data = pickle.load(f)
    result = {}
    with torch.no_grad():
        for tmp_test in tqdm(test_data):

            each_sub, _, author_id, pub_id  = tmp_test
            each_sub = each_sub.cuda()

            node_outputs, adj_matrix, adj_weight, batch_item = each_sub.x, each_sub.edge_index, each_sub.edge_attr.squeeze(-1), each_sub.batch

            # select feature 
            if args.threshold > 0:
                flag = adj_weight[:,1:]<args.threshold
                adj_weight[:,1:] = torch.where(flag,torch.tensor(0.0),adj_weight[:,1:])
            if args.usecoo and args.usecov:
                adj_weight = adj_weight.mean(dim = -1)
            elif args.usecoo:
                adj_weight = (adj_weight[:,0] + adj_weight[:,1])/2
            elif args.usecov:
                adj_weight = (adj_weight[:,0] + adj_weight[:,2])/2
            else:
                adj_weight = adj_weight[:,0]
            flag = torch.nonzero(adj_weight).squeeze(-1)
            adj_matrix = adj_matrix.T[flag].T
            adj_weight = adj_weight[flag]

            node_outputs, adj_weight, centroid, output_loss, centroid_loss, edge_prob = encoder(node_outputs, adj_matrix, adj_weight, batch_item, 1)

            centroid = centroid.squeeze(0)
            scores = criterion.get_score(node_outputs, centroid)
            
            result[author_id] = {}
            for i in range(len(pub_id)):
                result[author_id][pub_id[i]]=scores[i].item()
    
    with open(f'{args.saved_dir}/res.json', 'w') as f:
        json.dump(result, f)
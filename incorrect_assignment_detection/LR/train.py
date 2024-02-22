import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import LogisticRegressionModel
import pickle
import json
import random
from tqdm import tqdm
from utils import *
import os
train_dir = 'train_embedding.pkl'
eval_dir = 'eval_embedding.pkl'
test_dir = 'test_embedding.pkl'


with open(train_dir, 'rb') as files:
    train_data = pickle.load(files)

with open(eval_dir, 'rb') as files:
    eval_data = pickle.load(files)

with open(test_dir, 'rb') as files:
    test_data = pickle.load(files)

model = LogisticRegressionModel(input_dim=100).cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

if not os.path.exists("model"): # model paras
    os.makedirs("model")

max_map = -1
max_auc = -1
max_epoch = -1
early_stop_counter = 0
for epoch in range(500):
    model.train()
    batch_loss = []
    batch_labels = []
    batch_index = 0
    random.shuffle(train_data)
    for tmp_train in train_data:
        batch_index += 1
        features, label, author_id, pub_id = tmp_train
        features = features.cuda()
        label = label.float().cuda()
        logit = model(features)

        loss = criterion(logit.squeeze(), label)
        batch_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    avg_batch_loss = np.mean(np.array(batch_loss))
    print("Epoch:{} loss{:.6f}".format(epoch, avg_batch_loss))
    
    model.eval()
    label_list = []
    scores_list = []
    with torch.no_grad():
        for tmp_eval in eval_data:
            features, label, author_id, pub_id = tmp_eval
            features = features.cuda()

            logit = model(features)
            label_list.append(label.numpy())
            scores_list.append(logit.cpu().numpy())

    auc, maps = MAPs(label_list, scores_list)
    print("Epoch: {} Auc: {:.6f} Maps: {:.6f} Max-Auc: {:.6f} Max-Maps: {:.6f}".format(epoch, auc, maps, max_auc, max_map))
    if maps > max_map or auc > max_auc:
        early_stop_counter = 0

        max_epoch = epoch
        max_map = maps if maps > max_map else max_map
        max_auc = auc if auc > max_auc else max_auc

        torch.save(model, f"model/model_{str(epoch)}.pt")
        print("***************** Epoch: {} Max Auc: {:.6f} Maps: {:.6f} *******************".format(epoch,max_auc,max_map))
    else:
        early_stop_counter += 1
        if early_stop_counter >= 10:
            print("Early stop!")
            break
#test
model = torch.load(f"model/model_{str(max_epoch)}.pt")
result = {}
with torch.no_grad():
    for tmp_test in tqdm(test_data):
        features, label, author_id, pub_id = tmp_test
        features = features.cuda()
        logit = model(features)
        
        result[author_id] = {}
        for i in range(len(pub_id)):
            result[author_id][pub_id[i]]=logit[i].item()
with open('res.json', 'w') as f:
    json.dump(result, f, indent=4)
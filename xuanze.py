#encoding = utf-8
import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
from sklearn.metrics import accuracy_score
import numpy as np
import os
import shutil
import argparse
import sys
import logging
from datetime import datetime
from torch.optim import lr_scheduler
from model import BertGCN, BertGAT
import torch
import os
os.environ['NO_PROXY'] = 'stackoverflow.com'
import datetime
current_time = datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=36, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('-m', '--m', type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
parser.add_argument('--nb_epochs', type=int, default=200)
parser.add_argument('--bert_init', type=str, default='hfl/chinese-roberta-wwm-ext',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased','hfl/chinese-roberta-wwm-ext'])
parser.add_argument('--pretrained_bert_ckpt', default=None)
parser.add_argument('--dataset', default='微博', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr','微博'])
parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat'])
parser.add_argument('--gcn_layers', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=200, help='the dimension of gcn hidden layer, the dimension for gat is n_hidden * heads')
parser.add_argument('--heads', type=int, default=8, help='the number of attentionn heads for gat')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--gcn_lr', type=float, default=5e-5)
parser.add_argument('--bert_lr', type=float, default=6e-5)
args = parser.parse_args()
max_length = args.max_length
batch_size = args.batch_size
m = args.m
nb_epochs = args.nb_epochs
bert_init = args.bert_init
pretrained_bert_ckpt = args.pretrained_bert_ckpt
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir
gcn_model = args.gcn_model
gcn_layers = args.gcn_layers
n_hidden = args.n_hidden
heads = args.heads
dropout = args.dropout
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr


ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
cpu = th.device('cpu')
gpu = th.device('cuda:0')
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_node = features.shape[0]
test_idx = Data.TensorDataset(th.arange(0, nb_node, dtype=th.long))
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)

nb_class = y_train.shape[1]


model = BertGCN(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    n_hidden=n_hidden, dropout=dropout)
model.eval()#注意这个
state_dict = torch.load(os.path.join(ckpt_dir, 'checkpoint.pth'))
    # name = k[2:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
    # print(name)
model.load_state_dict(state_dict)
def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
    return input.input_ids, input.attention_mask
corpse_file = './data/corpus/' + dataset +'_shuffle.txt'
input_ids, attention_mask = {}, {}
with open(corpse_file, 'r',encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\\', '')
    text = text.split('\n')


nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
input_ids, attention_mask = encode_input(text, model.tokenizer)
input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
attention_mask = th.cat([attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])




y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
g.ndata['label_train'] = th.LongTensor(y_train)
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

doc_mask  = train_mask + val_mask + test_mask
def update_feature():
    global model, g, doc_mask
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=1024
    )
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    g = g.to(cpu)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g
g = update_feature()
g.ndata['cls_feats'].detach_()

i = 0
def test_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(g, idx)

        global i
        i+=1
        print(i)
        y_true = g.ndata['label'][idx]
        return y_pred, y_true
evaluator = Engine(test_step)



metrics = evaluator.state.metrics

metrics={
    'acc': Accuracy()[0],
    'right':Accuracy()[1],
    'nll': Loss(th.nn.NLLLoss())
}#中文
for n, f in metrics.items():
    f.attach(evaluator, n)
evaluator.run(idx_loader_test)
metrics = evaluator.state.metrics
test_acc, test_nll,test_right = metrics["acc"], metrics["nll"],metrics["right"]
print(test_acc,test_right)
test_right_list = []
for index,k in enumerate(test_right):
    for i in k:
        test_right_list.append(i+index*batch_size)
print(test_right_list)
print(len(test_right_list))
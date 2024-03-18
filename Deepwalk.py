import argparse
import os.path as osp
import time
import networkx as nx
import random
from gensim.models.word2vec import Word2Vec
import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--steps_per_walk', type=int, default=40)
parser.add_argument('--walks_per_node', type=int, default=80)
parser.add_argument('--window_size', type=int, default=10)
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

init_wandb(
    name=f'DeepWalk-{args.dataset}',
    epochs=args.epochs,
    lr=args.lr,
    device=device,
)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0]

edge_index = data.edge_index
edges = edge_index.t().numpy()
G = nx.from_edgelist(edges, create_using=nx.Graph())

num_graph_node = len(data['x'])


class Model(nn.Module):
    def __init__(self, num_graph_node, dim) -> None:
        super().__init__()
        self.phi = nn.Parameter(torch.rand(num_graph_node, dim), requires_grad=True)
        self.phi2 = nn.Parameter(torch.rand(num_graph_node, dim),requires_grad=True)
    
    def forward(self, vtx):
        x = torch.matmul(vtx, self.phi)
        out = torch.matmul(self.phi, x)

        return out
        
model = Model(num_graph_node, args.dim)

optimizer = torch.optim.SGD(model.parameters(), args.lr)

def skip_gram(rw_seq, window_size):
    for seq_idx, v_j in enumerate(rw_seq):
        front, rear = seq_idx - window_size//2, seq_idx + window_size // 2
        front, rear = max(0, front), min(len(rw_seq)-1, rear)
        one_hot = torch.zeros(num_graph_node)
        one_hot[v_j] = 1
        out = model(one_hot)

        for idx in range(front, rear):
            optimizer.zero_grad()
            vtx = rw_seq[idx]
            loss = torch.log(torch.sum(torch.exp(out))) - out[vtx]
            loss.backward(retain_graph = True)
            optimizer.step()

            
    for j in range(len(wvi)):
        for k in range(max(0,j- window_size) , min(j+ window_size, len(wvi))):
            one_hot = torch.zeros(num_graph_node)
            one_hot[wvi[j]] = 1
            out = model(one_hot)
            loss = torch.log(torch.sum(torch.exp(out))) - out[wvi[k]]
            loss.backward()
            
            for param in model.parameters():
                try:
                    param.grad.data.zero_()
                    param.data.sub_(args.lr*param.grad)
                except:
                    pass

def random_walk(node, G):
    walk = [node]
    for _ in range(args.walks_per_node - 1):
        neighbors = list(G.neighbors(walk[-1]))
        if len(neighbors) == 0:
            break
        walk.append(random.choice(neighbors))
    return walk


for i in range(args.walks_per_node):
    vtx = list(G.nodes)
    random.shffule(vtx)

    for vi in vtx:
        wvi = random_walk(vi,G)
        skip_gram(wvi, args.window_size)
        




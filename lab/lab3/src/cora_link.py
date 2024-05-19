import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

torch.manual_seed(42)

# 辅助函数：将edge_index转换为邻接矩阵
def edge_index_to_adjacency_matrix(edge_index):
    num_nodes = edge_index.max().item() + 1
    adjacency_matrix = torch.zeros(num_nodes, num_nodes)
    edge_index_t = edge_index.t()
    adjacency_matrix[edge_index_t[0], edge_index_t[1]] = 1
    return adjacency_matrix

# 定义图卷积层
class GCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)  # Sparse matrix multiplication
        return output

# 定义PairNorm层，用于对输入进行归一化
class PairNorm(nn.Module):
    def __init__(self, scale: float = 1, eps: float = 1e-5):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=0, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.scale * x
        return x

# 定义链接预测模型
class LinkPrediction(nn.Module):
    def __init__(self, num_features, hidden_channels, edge_index):
        super(LinkPrediction, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        adj = edge_index_to_adjacency_matrix(edge_index)
        adj = adj + torch.eye(adj.size(0))
        D = torch.diag(torch.pow(adj.sum(1).float(), -0.5))
        A = torch.mm(torch.mm(D, adj), D)
        self.A = A
        self.norm = PairNorm()

    def forward(self, x, edge_index):
        x = self.norm(x)
        x = self.conv1(x, self.A)
        x = x.relu()
        x = self.norm(x)
        x = self.conv2(x, self.A)
        x = x.relu()
        x = self.norm(x)
        x = self.conv3(x, self.A)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

# 训练步骤
def train(model, optimizer, criterion, data, pos_edge_index, neg_edge_index):
    model.train()
    optimizer.zero_grad()
    num_edges = int(data.edge_index.size(1) * 0.5)
    edge_index = data.edge_index[:, :num_edges]
    z = model(data.x, edge_index)
    logits = model.decode(z, pos_edge_index, neg_edge_index)
    labels = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).to(data.x.device)
    loss = criterion(logits[data.train_mask], labels[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

# 验证步骤
def val(model, data, neg_edge_index, criterion):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        logits = model.decode(z, data.edge_index, neg_edge_index)
        labels = torch.cat([torch.ones(data.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).to(data.x.device)
        loss = criterion(logits[data.val_mask], labels[data.val_mask])
        preds = torch.sigmoid(logits[data.val_mask]).detach().cpu().numpy()
        labels = labels[data.val_mask].detach().cpu().numpy()
        auc = roc_auc_score(labels, preds)
    return loss, auc

# 测试步骤
def test(model, data, neg_edge_index, criterion):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        logits = model.decode(z, data.edge_index, neg_edge_index)
        labels = torch.cat([torch.ones(data.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).to(data.x.device)
        loss = criterion(logits[data.test_mask], labels[data.test_mask])
        preds = torch.sigmoid(logits[data.test_mask]).detach().cpu().numpy()
        labels = labels[data.test_mask].detach().cpu().numpy()
        auc = roc_auc_score(labels, preds)
    print(f'Test Loss: {loss}, Test AUC: {auc}')
    return auc

# 主训练函数
def train_link_prediction(model, data, optimizer, criterion, num_epochs=100):
    train_loss_arr, val_loss_arr, auc_arr = [], [], []
    for epoch in range(num_epochs):
        neg_edge_index = negative_sampling(data.edge_index, num_neg_samples=data.edge_index.size(1))
        loss = train(model, optimizer, criterion, data, data.edge_index, neg_edge_index)
        loss2, auc = val(model, data, neg_edge_index, criterion)
        print(f'Epoch: {epoch}, Train Loss: {loss}, Val Loss: {loss2}, Val AUC: {auc}')
        train_loss_arr.append(loss.detach().numpy())
        val_loss_arr.append(loss2.detach().numpy())
        auc_arr.append(auc)
    return train_loss_arr, val_loss_arr, auc_arr

# 加载数据集
cora = Planetoid(".", "Cora", split='geom-gcn')
data = cora[0]

# 创建模型
num_edges = int(data.edge_index.size(1) * 0.5)
edge_index = data.edge_index[:, :num_edges]
model = LinkPrediction(cora.num_features, 64, edge_index)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

# 划分数据集
num_edges = 2 * data.edge_index.size(1)
edge_indices = np.arange(num_edges)
train_indices, rest_indices = train_test_split(edge_indices, test_size=0.3, random_state=42)
val_indices, test_indices = train_test_split(rest_indices, test_size=0.5, random_state=42)
train_mask = torch.zeros(num_edges, dtype=torch.bool)
val_mask = torch.zeros(num_edges, dtype=torch.bool)
test_mask = torch.zeros(num_edges, dtype=torch.bool)
train_mask[train_indices] = True
val_mask[val_indices] = True
test_mask[test_indices] = True
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# 训练模型
train_loss_arr, val_loss_arr, auc_arr = train_link_prediction(model, data, optimizer, criterion)

# 绘制损失曲线和AUC曲线
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

axs[0].plot(train_loss_arr, label='Train Loss')
axs[0].plot(val_loss_arr, label='Validation Loss')
axs[0].set_title('Train and Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(auc_arr, label='Validation AUC', color='orange')
axs[1].set_title('Validation AUC over Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('AUC')
axs[1].legend()

plt.tight_layout()
plt.savefig('cora_link.png')

# 测试模型
test_auc = test(model, data, negative_sampling(data.edge_index, num_neg_samples=data.edge_index.size(1)), criterion)
print(f'Test AUC: {test_auc}')

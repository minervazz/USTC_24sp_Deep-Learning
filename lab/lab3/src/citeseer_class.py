import numpy as np  
import os  

import torch  
from torch import nn  
from torch.nn import Softmax  
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix  

from torch_geometric.datasets import Planetoid  # Datasets for graph-based learning
from torch_geometric.transforms import RandomNodeSplit
import matplotlib.pyplot as plt
from torch_geometric.utils import add_self_loops

torch.manual_seed(42) # 设置随机种子

# 加载Cora数据集
cora = Planetoid( 
    ".", 
    "CiteSeer",
    split='geom-gcn', 
    transform=RandomNodeSplit( 
        num_val=300,
        num_test=500,
    )
)

# 定义图卷积层
class GraphConv(nn.Module): 
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 计算归一化系数
        row, col = edge_index
        deg = self.degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # inplace inf by 0
        norm[torch.isinf(norm)] = 0
        # Compute the adjacency matrix
        adj = torch.sparse_coo_tensor(edge_index, norm, (x.size(0), x.size(0)), dtype=x.dtype, device=x.device)

        # Perform the linear transformation
        x = self.linear(x)

        # Perform the convolution
        return adj @ x

    @staticmethod # 静态方法
    def degree(index: torch.Tensor, num_nodes: int, dtype: torch.dtype = None) -> torch.Tensor:
        # 计算每个节点的度
        out = torch.zeros(num_nodes, dtype=dtype, device=index.device)
        out.scatter_add_(0, index, out.new_ones((index.size(0),)))
        return out

# 定义PairNorm层，用于对输入进行归一化
class PairNorm(nn.Module):
    # Pairwise normalization layer.
    def __init__(self, scale: float = 1, eps: float = 1e-5):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=0, keepdim=True)

        # Normalize the input tensor
        x = (x - mean) / torch.sqrt(var + self.eps)

        # Scale the normalized tensor
        x = self.scale * x

        return x

# 定义GCN模型，三个图卷积层和一个Softmax分类层
class GCN(nn.Module):
    def __init__(self):
        super().__init__()        
        # 第一层图卷积层
        self.conv1 = GraphConv(cora.num_features, 500)
        self.norm = PairNorm() 
        # 第二层图卷积层
        self.conv2 = GraphConv(500, 100)        
        # 第三层图卷积层，输出维度为类别数
        self.conv3 = GraphConv(100, cora.num_classes)        
        # 定义softmax分类器, 将输入向量的每个元素缩放到（0,1）区间，使得整个向量的和为1
        self.classifier = Softmax(dim=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x = self.norm(x)        
        h = self.conv1(x, edge_index)
        h= h.tanh()
        # h = self.norm(h)
        
        # 第二层
        h = self.conv2(h,edge_index)
        h = h.tanh()
        # h = self.norm(h)
        
        # 第三层
        h = self.conv3(h,edge_index)
        h=h.tanh()
        
        # 分类器
        h = self.classifier(h)
        
        return h

# 训练步骤
def train(model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, data: torch.Tensor) -> tuple:
    model.train()
    
    # 梯度清零
    optimizer.zero_grad() 
    
    # 前向传播
    out = model(data.x, data.edge_index)
    
    # 计算训练loss
    train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # 反向传播
    train_loss.backward()
    
    # 更新优化器
    optimizer.step()
    
    # 计算验证集loss
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    
    # 计算验证集准确率
    predicted_labels = out.argmax(dim=1)[data.val_mask]
    acc_metric = MulticlassAccuracy()
    acc_metric.update(predicted_labels, data.y[data.val_mask])
    
    return train_loss, val_loss, acc_metric.compute()

# 评估函数
def eval(model: nn.Module, data: torch.Tensor, data_mask: torch.Tensor) -> torch.Tensor:
    model.eval()
    
    # 禁用梯度
    with torch.no_grad():
        out = model(data.x, data.edge_index) 
    
    # 预测标签
    predicted_labels = out.argmax(dim=1)[data_mask]
    
    # 计算准确率
    y = data.y[data_mask]
    acc_metric = MulticlassAccuracy()
    acc_metric.update(predicted_labels, y)
    
    return acc_metric.compute()

# 计算混淆矩阵
def compute_confusion_matrix(model: nn.Module, data: torch.Tensor) -> torch.Tensor:
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    
    predicted_labels = out.argmax(dim=1)
    cf_metric = MulticlassConfusionMatrix(cora.num_classes)
    cf_metric.update(predicted_labels, data.y)
    
    return cf_metric.compute()

# 随机丢弃边
def drop_edge(edge_index, drop_prob): 
    if drop_prob < 0 or drop_prob > 1:
        raise ValueError("`drop_prob` must be between 0 and 1")

    # 要删除的边的数量
    num_edges = edge_index.size(1)
    num_edges_drop = int(drop_prob * num_edges)

    # 创建一个掩码用于删除边
    mask = torch.ones(num_edges, dtype=torch.bool)
    drop_indices = torch.randperm(num_edges)[:num_edges_drop]
    mask[drop_indices] = False

    return edge_index[:, mask]

# 训练模型
def train_cora(test_pc: float, epochs: int) -> tuple:
    num_test = int(cora[0].num_nodes * test_pc)
    num_val = int(cora[0].num_nodes * 0.1) # 验证集比例
    
    # 重新加载数据集
    dataset = Planetoid(
        ".",
        "Citeseer",
        split='geom-gcn',
        transform=RandomNodeSplit(
            num_val=num_val,
            num_test=num_test,
        )
    )
    data = dataset[0]

    # 初始化模型、优化器和损失函数
    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_loss_arr = []
    val_loss_arr = []
    test_loss_arr = []  
    train_acc_arr = []
    val_acc_arr = []
    test_acc_arr = []
    
    # 早停
    patience = 0
    epsilon = 0.0001
    min_val_loss = float('inf')  # 确保第一个epoch一定会更新
    
    for epoch in range(epochs):
        raw_edge_index = data.edge_index
        data.edge_index = drop_edge(data.edge_index, 0.04) # dropedge,设置为0.04
        train_loss, val_loss, val_acc = train(model, optimizer, criterion, data)
        
        train_loss_arr.append(train_loss.detach().numpy())
        
        # 早停
        if epoch == 0 or (min_val_loss - val_loss) > epsilon:
            min_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            print(f'Early stopping patience counter: {patience}') # 早停标记
            if patience >= 5:
                print(f'Early stopping triggered at epoch {epoch}')
                break
        
        val_loss_arr.append(val_loss.detach().numpy())
        
        # 计算训练集准确率
        train_acc = eval(model, data, data.train_mask)
        train_acc_arr.append(train_acc)
        
        # 计算验证集准确率
        val_acc_arr.append(val_acc)
        
        # 计算测试集loss
        test_loss = criterion(model(data.x, data.edge_index)[data.test_mask], data.y[data.test_mask])
        test_loss_arr.append(test_loss.detach().numpy())
        test_acc = eval(model, data, data.test_mask)
        test_acc_arr.append(test_acc)
        
        print(f'Epoch: {epoch} \t Training loss: {train_loss} \t Validation loss: {val_loss} \t Validation Accuracy: {val_acc}')
        data.edge_index = raw_edge_index # 恢复原始边索引   
    
    # 最终在测试集上评估模型
    test_acc = eval(model, data, data.test_mask)
    test_cf = compute_confusion_matrix(model, data)
    
    return train_loss_arr, val_loss_arr, test_loss_arr, train_acc_arr, val_acc_arr, test_acc_arr, test_acc, test_cf

# 训练模型
train_loss_arr, val_loss_arr, test_loss_arr, train_acc_arr, val_acc_arr, test_acc_arr, test_acc, test_cf = train_cora(0.2, 100) 

# 输出结果
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# 训练结果及验证结果
ax[0].plot(train_loss_arr, label='Training loss')
ax[0].plot(val_loss_arr, label='Validation loss')

ax[0].legend()
ax[0].set_title('Loss Curves', fontsize=20)
ax[0].set_xlabel('Epochs')

# 输出准确率
ax[1].plot(train_acc_arr, label='Training Accuracy')
ax[1].plot(val_acc_arr, label='Validation Accuracy')

ax[1].legend()
ax[1].set_title('Accuracy Curves', fontsize=20)
ax[1].set_xlabel('Epochs')

# 输出混淆矩阵
ax[2].imshow(test_cf.numpy(), cmap='viridis')
ax[2].set_title('Multiclass Confusion Matrix', fontsize=20)

plt.savefig('citeseer_class.png')

# 打印最终测试集准确率
print(f'Test Accuracy: {test_acc}')

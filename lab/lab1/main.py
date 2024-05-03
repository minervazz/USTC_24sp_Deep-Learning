# 导入必要的库
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 定义目标函数
def target_function(x):
    return np.log2(x) + np.cos(np.pi * x / 2)

# 生成和预处理数据
x = np.linspace(1, 16, num=10000)
y = target_function(x) 
x_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.float32) # 将x转换为张量
y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32) # 将y转换为张量
X_train, X_temp, y_train, y_temp = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42) # 划分训练集和剩余数据
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # 划分验证集和测试集

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 256)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(256, 256) # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(256, 256) # 新增加 第二个隐藏层到第三个隐藏层
        self.fc4 = nn.Linear(256, 1)  # 第三个隐藏层到输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一个隐藏层激活
        x = F.relu(self.fc2(x))  # 第二个隐藏层激活
        x = F.relu(self.fc3(x))  # 第三个隐藏层激活
        return self.fc4(x)  # 输出层，没有激活函数

model = Net()

# 训练模型
criterion = nn.MSELoss() # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.009) # 定义优化器


for epoch in range(15000):
    optimizer.zero_grad() # 梯度清零
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward() # 反向传播
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# # 模型评估
# model.eval()
# with torch.no_grad():
#     predictions = model(X_val).numpy()

# # 绘制验证集实际值和预测值
# plt.figure(figsize=(10, 6))
# plt.scatter(X_val.numpy(), y_val.numpy(), color='blue', label='Actual values')
# plt.scatter(X_val.numpy(), predictions, color='red', label='Predicted values')
# plt.title('Comparison of Actual and Predicted Values on Validation Set')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend()
# plt.show()

# 模型评估 - 使用测试集
model.eval()  # 将模型设置为评估模式
with torch.no_grad():  # 关闭梯度计算
    predictions_test = model(X_test)  # 在测试集上进行预测
    test_loss = criterion(predictions_test, y_test)  # 计算测试集上的损失
    print(f'Test Loss: {test_loss.item():.4f}')  # 打印测试集损失

# 可视化测试集的实际值和预测值
plt.figure(figsize=(10, 6))
plt.scatter(X_test.numpy(), y_test.numpy(), color='blue', label='Actual values')
plt.scatter(X_test.numpy(), predictions_test.numpy(), color='red', label='Predicted values')
plt.title('Test Set: Actual vs Predicted Values')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

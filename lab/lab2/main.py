import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

# 数据预处理操作，包括数据增强技术
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 数据增强：图像随机水平翻转
    transforms.RandomCrop(32, padding=4),  # 数据增强：随机裁剪
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化
])

# 加载数据集
data_root = './data'
train_data = datasets.CIFAR10(root=data_root, train=True, transform=transform, download=False) # 训练集
test_data = datasets.CIFAR10(root=data_root, train=False, transform=transform, download=False) # 测试集

# 划分训练集和验证集
train_size = int(0.8 * len(train_data)) # 现训练集size
val_size = len(train_data) - train_size # 现验证集size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size]) # 原始数据随机划分

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False) 
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 3x3的卷积核，边缘填充1个像素
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(64) # 批标准化
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # 池化层，2x2的窗口进行池化操作，步长为2
        self.pool = nn.MaxPool2d(2, 2) 

        # 全连接层
        self.fc1 = nn.Linear(256 * 4 * 4, 1024) 
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.3)  # Dropout 概率 0.3

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) 
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# 实例化模型并移至GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=3, gamma=0.8)  # 学习率衰减 learning rate decay

# 训练模型函数
def train_model(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, device):
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        val_accuracy = validate_model(model, val_loader, device) # 每轮训练结束后，验证模型
        val_accuracies.append(val_accuracy)

        scheduler.step()  # 更新学习率
        print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    return train_losses, val_accuracies

# 验证模型函数
def validate_model(model, val_loader, device):
    model.eval()
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    
    accuracy = total_correct / total_images * 100
    return accuracy

# 调用训练模型函数
train_losses, val_accuracies = train_model(14, model, train_loader, val_loader, criterion, optimizer, scheduler, device)

# 绘制损失和准确率曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, 15), train_losses, label='Train Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, 15), val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()

# 测试模型函数
def test_model(model, test_loader, device):
    model.eval()  # 设置模型为评估模式
    total_correct = 0
    total_images = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()  # 重新声明损失函数，用于计算测试损失

    with torch.no_grad():  # 不追踪梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = (total_correct / total_images) * 100
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

# 调用测试模型函数
test_loss, test_accuracy = test_model(model, test_loader, device)
print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.2f}%")


"""
log:
1. 0.35 的 drop out, 学习率衰减无效, 感觉每3周期验证集准确率会停滞, 可以减小学习率
2. 0.3 的 drop out, 学习率衰减每3 epoch衰减 0.8, 学习率初始化0.001
3. 0.32 的 drop out, 学习率衰减不变, 学习率初始化0.001, 增加epoch到14, 因为这样的话刚好衰减了4次
4. 0.3 的 drop out, 学习率衰减不变, 学习率初始化0.001, epoch 14 (因为发现还是第二组最好), 不过后期依然出现过拟合
5. 0.3 的 drop out, 学习率衰减改为 0.78, 学习率初始化0.001, epoch 14

网络深度和核数, 根据问题形态选择3与3*3。
由于这个设置已经可以满足80%的准确率, 为了不增加模型复杂程度, 不再进行更复杂的设计。
"""
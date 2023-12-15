import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import load_iris

iris = load_iris()

# 入力値と目標値を抽出
x = iris["data"]
t = iris["target"]

# print(iris)

# print(type(x))
# print(type(t))

# pythonのtensor型へ変換
x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.int64)

# print(type(x))
# print(type(t))

# print(x.shape)
# print(t.shape)

# 入力値と目標値を纏める
dataset = torch.utils.data.TensorDataset(x, t)

# 各データのサンプル数を決定
# train:val:test=60%:20%:20%
n_train = int(len(dataset) * 0.6)
n_val = int(len(dataset) * 0.2)
n_test = len(dataset) - n_train - n_val

torch.manual_seed(0)

# データセットの分割
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

# print(len(train))
# print(len(val))
# print(len(test))

# ミニバッチ学習

# バッチサイズの定義
batch_size = 10
train_loader = torch.utils.data.DataLoader(
    train, batch_size, shuffle=True, drop_last=True
)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)

x, t = next(iter(train_loader))

# print(x)
# print(t)


# ネットワークの定義
# 4⇒４⇒３の全結合層を定義
class Net(nn.Module):
    # 使用するオブジェクトを定義
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 3)

    # 順伝播
    def forward(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)
        return h


torch.manual_seed(0)
# インスタンス化
net = Net()
# print(net)

opitimizer = torch.optim.SGD(net.parameters(), lr=0.01)

batch = next(iter(train_loader))

x, t = batch

# 予測値の算出
y = net.forward(x)

loss = F.cross_entropy(y, t)

loss.backward()

opitimizer.step()

flg = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

net.to(device)

x = x.to(device)

t = t.to(device)
# 勾配情報の初期化
opitimizer.zero_grad()

# エポック数
max_epoch = 1
# モデルの初期化
torch.manual_seed(0)
# モデルのインスタンス化とデバイスへの転送
net = Net().to(device)

# 最適化手法
opitimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 学習のループ
for epoch in range(max_epoch):
    for batch in train_loader:
        x, t = batch
        x = x.to(device)
        t = t.to(device)
        y = net(x)
        loss = F.cross_entropy(y, t)
        print(f"loss:{loss}")
        opitimizer.zero_grad()
        loss.backward()
        opitimizer.step()

# 評価指標の追加
x, t = next(iter(train_loader))
x = x.to(device)
t = t.to(device)
y = net(x)
# print(f"評価指標の追加:{y}")

y_label = torch.argmax(y, dim=1)

# print(f"y_label:{y_label}")
# 正解率
accuracy = (y_label == t).sum().float() / len(t)
print(f"accuracy:{accuracy}")

# モデルの初期化
torch.manual_seed(0)
# モデルのインスタンス化とデバイスへの転送
net = Net().to(device)

# 最適化手法
opitimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 学習のループ
for epoch in range(max_epoch):
    for batch in train_loader:
        x, t = batch
        x = x.to(device)
        t = t.to(device)
        y = net(x)
        loss = F.cross_entropy(y, t)
        # 正解率追加
        y_label = torch.argmax(y, dim=1)
        accuracy = (y_label == t).sum().float() / len(t)
        # print(f"accuracy:{accuracy:.2f}")
        opitimizer.zero_grad()
        loss.backward()
        opitimizer.step()


def calc_accuracy(data_loader):
    with torch.no_grad():
        total = 0
        correct = 0.0
        for batch in data_loader:
            x, t = batch
            x = x.to(device)
            t = t.to(device)
            y = net(x)
            y_label = torch.argmax(y, dim=1)
            total += len(t)
            correct += (y_label == t).sum()

        # 全体の平均を算出
        accuracy = correct / total
    return accuracy


accuracy = calc_accuracy(val_loader)

print(f"accuracy:{accuracy:.2f}")

accuracy = calc_accuracy(test_loader)

print(f"accuracy:{accuracy:.2f}")

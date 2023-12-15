import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from sklearn.datasets import load_iris

# データセットの読み込み
iris = load_iris()

# 入力値と目標値を抽出
x = iris["data"]
t = iris["target"]

# pythonのtensor型へ変換
x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.int64)


# 入力値と目標値を纏める
dataset = torch.utils.data.TensorDataset(x, t)

# 各データのサンプル数を決定
# train:val:test=60%:20%:20%
n_train = int(len(dataset) * 0.6)
n_val = int(len(dataset) * 0.2)
n_test = len(dataset) - n_train - n_val

# データセットの分割
# ランダムに分割を行うため、シードを固定して再現性を確保
torch.manual_seed(0)
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])


# バッチサイズの定義
batch_size = 10

# Data　Loaderを用意
# shuffle はデフォルトでFalseのため、訓練データのみ　Trueに指定
train_loader = torch.utils.data.DataLoader(
    train, batch_size, shuffle=True, drop_last=True
)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)


# ネットワークの定義


# 4⇒４⇒３の全結合層を定義
class Net(pl.LightningModule):
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

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer


pl.seed_everything(0)
# インスタンス化
net = Net()
# 学習を行う　Trainer
trainer = pl.Trainer(max_epochs=30, deterministic=True)

# 学習の実行
trainer.fit(net, train_loader)

# print(trainer.fit(net, train_loader))

trainer.callback_metrics


results = trainer.test(dataloaders=test_loader)
print(trainer.callback_metrics)
print(results)

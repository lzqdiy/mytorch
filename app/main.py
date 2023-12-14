import torch
import torch.nn as nn
import torch.nn.functional as F

# 乱数のシードを固定
torch.manual_seed(0)

print(torch.__version__)

# 3 ノード⇒２ノードの全結合層（fully-connected layer）
fc = nn.Linear(3, 2)

print(fc)


# 重み
weight = fc.weight
print(f"重み:{weight}")

# バイアス
bias = fc.bias
print(f"バイアス:{bias}")

# 線形変換
# データ型をテンソル型に変える
x = torch.Tensor([[1., 2., 3]])

print(type(x))

print(x.dtype)

# 線形変換の計算
u = fc(x)

print(f"u:{u}")

# 非線形変換の計算
# ReLU関数
h = F.relu(u)

print(f"h:{h}")

# 目的関数
# 目標値
t = torch.Tensor([[1.], [3.]])

# 予測値
y = torch.Tensor([[2.], [4.]])

# 平均二重誤差の算出
ty=F.mse_loss(y,t)

print(f"ty:{ty}")

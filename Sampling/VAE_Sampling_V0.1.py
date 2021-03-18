import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 数据集类
class MyDataset(Dataset):

    # Initialization
    def __init__(self, data, mode='2D'):
        self.data, self.mode = data,  mode

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :]
        elif self.mode == '3D':
            return self.data[:, index, :]

    # Get length
    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[1]

class VAE(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)

        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)

    def encode(self, X):
        H1 = torch.sigmoid(self.fc1(X))
        return self.fc21(H1), self.fc22(H1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, Z):
        H = torch.sigmoid(self.fc3(Z))
        return torch.sigmoid(self.fc4(H))

    def forward(self, inputs, generate=False):
        # 训练模式
        if generate is False:
            mu, logvar = self.encode(inputs)
            z = self.reparameterize(mu, logvar)

            x_reconst = self.decode(z)
            return x_reconst, mu, logvar
        # 生成模式
        else:
            x_generate = self.decode(inputs)
            return x_generate


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: 重构的数据
    x: 原始的数据
    mu: 潜空间变量均值
    logvar: 潜空间变量方差
    """

    # MSE误差
    MSE = F.mse_loss(recon_x, x, reduction='mean')

    # 先验概率设置为标准正态分布
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KL divergence

    KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE, KLD, (MSE + KLD)

class VAE_Sampling(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, h_dim, z_dim, lr=0.001, device='cuda:0', n_epoch=200, batch_size=64, seed=1024):
        super(VAE_Sampling, self).__init__()

        # Set seed
        torch.manual_seed(seed)

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.device = device

        # 模型实例化
        self.VAE_Model = VAE(input_dim=input_dim, h_dim=h_dim, z_dim=z_dim).to(device)
        self.optimizer = optim.Adam(self.VAE_Model.parameters(), lr=lr)
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.loss_hist = []
        self.MSE_hist = []
        self.KLD_hist = []

        self.seed = seed


        # Initialize scaler
        # 这里可以自己再进行适当的改动
        #self.scaler_X = StandardScaler()
        self.scaler_X = MinMaxScaler()
    # 训练模型
    def fit(self, X):
        X = self.scaler_X.fit_transform(X)

        dataset = MyDataset(torch.tensor(X, dtype=torch.float32, device=self.device), mode='2D')
        self.VAE_Model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            self.MSE_hist.append(0)
            self.KLD_hist.append(0)

            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X in data_loader:



                x_reconst, mu, logvar = self.VAE_Model(inputs=batch_X, generate=False)

                MSE, KLD, loss = loss_function(recon_x=x_reconst, x=batch_X, mu=mu, logvar=logvar)


                self.loss_hist[-1] += loss.item()
                self.MSE_hist[-1] += MSE.item()
                self.KLD_hist[-1] += KLD.item()

                self.VAE_Model.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('Epoch: {}, Loss: {}'.format(i + 1, self.loss_hist[-1]))
            print('MSE损失大小是: {}, KL散度损失大小是: {}'.format(self.MSE_hist[-1], self.KLD_hist[-1]))
        print('Optimization finished!')
        return self

    # 实操数据
    def generate(self, sample_number=1000):
        
        # 卡住随机种子编号, 保证结果重复性
        np.random.seed(self.seed)
        # 判别到底是一维还是多维的
        if self.z_dim is 1:
            X = np.random.normal(loc=0.0, scale=1.0, size=sample_number)
        else:
            mean = np.zeros(self.z_dim)
            # mean = np.array([self.z_dim, 0])
            cov = np.eye(self.z_dim)
            X = np.random.multivariate_normal(mean=mean, cov=cov, size=sample_number)

        # 放上torch框架
        X = torch.tensor(X, dtype=torch.float32, device=self.device)

        # 重构维度确保没出岔子
        X = X.view(sample_number, self.z_dim)

        with torch.no_grad():
            # 生成数据
            X_generate =self.VAE_Model(inputs=X, generate=True).cpu().numpy()
            # 生成结果
            X_generate = self.scaler_X.inverse_transform(X_generate)

        print('Scenario generate finished')

        return X_generate

# 数据读取及其对应的实操
# 这里的名字改成你要的数据的名字
data = pd.read_csv('Debutanizer_Data.txt', sep = '\s+')
# 转为numpy
X_raw = np.array(data)
# lr学习率可以适当调整,设置seed随机种子是为了让结果具有重复性
Sampling = VAE_Sampling(input_dim=data.shape[1], h_dim=2, z_dim=1, n_epoch=150, device='cuda:0', lr=0.01).fit(X_raw)
# sample number是进行采样的数据多少
X_generate = Sampling.generate(sample_number=1000)
# 写入csv,到时候转为excel
np.savetxt('VAE_Sampling.csv', X_generate, delimiter=',')

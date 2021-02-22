import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

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

# Encoder
class Q_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Q_Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

    def forward(self, X):
        X = self.act(self.fc1(X))
        X = self.act(self.fc2(X))

        # Gaussian to judge
        X = self.fc3(X)

        return X

# Decoder
class P_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(P_Net, self).__init__()
        self.fc1 = nn.Linear(output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

        self.act = nn.ReLU()


    def forward(self, X):
        X = self.act(self.fc1(X))
        X = self.act(self.fc2(X))

        # to input
        X = self.fc3(X)

        return torch.sigmoid(X)

# Discriminator
class D_Net_Gauss(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(D_Net_Gauss, self).__init__()
        self.fc1 = nn.Linear(output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.act = nn.ReLU()

    def forward(self, X):
        X = self.act(self.fc1(X))
        X = self.act(self.fc2(X))

        X = self.fc3(X)

        return torch.sigmoid(X)


class AAEModel():
    def __init__(self, input_dim, hidden_dim, z_dim, gen_lr, reg_lr, EPS=1.0e-15, batch_size=16, device='cuda:0', seed=1024, num_epochs=300):
        super(AAEModel, self).__init__()

        # 超参数继承
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.gen_lr = gen_lr
        self.reg_lr = reg_lr
        # EPS是为了防止对数运算出现数值问题
        self.EPS = EPS
        self.batch_size = batch_size

        self.device = device
        self.seed = seed

        self.num_epochs = num_epochs

        torch.manual_seed(seed)

        # 模型
        self.encoder = Q_Net(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=z_dim).to(device)
        self.decoder = P_Net(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=z_dim).to(device)

        self.discriminator = D_Net_Gauss(hidden_dim=hidden_dim, output_dim=z_dim).to(device)

        # 优化器(编码器部分)
        self.optim_decoder = torch.optim.Adam(self.decoder.parameters(), lr=gen_lr)
        self.optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=gen_lr)

        # 优化器(判别器部分)
        self.optim_generator = torch.optim.Adam(self.encoder.parameters(), lr=reg_lr)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=reg_lr)

        # 损失函数列表
        self.reconst_loss_hist = []
        self.D_loss_hist = []
        self.G_loss_hist = []

    def fit(self, X):

        dataset = MyDataset(torch.tensor(X, dtype=torch.float32, device=self.device), mode='2D')

        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        for epoch in range(self.num_epochs):
            self.reconst_loss_hist.append(0)
            self.D_loss_hist.append(0)
            self.G_loss_hist.append(0)

            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X in data_loader:

                '------------------ 自编码器部分 ------------------'
                # 梯度清零
                self.encoder.zero_grad()
                self.decoder.zero_grad()

                # 编码与解码
                z_samples = self.encoder(batch_X)
                X_samples = self.decoder(z_samples)

                # 重构误差计算(MSE Loss)
                reconst_loss = torch.nn.functional.mse_loss(input=X_samples, target=batch_X, reduction='mean')

                self.reconst_loss_hist[-1] += reconst_loss.item()

                reconst_loss.backward()



                self.optim_encoder.step()
                self.optim_decoder.step()

                '------------------ GAN网络部分 ------------------'
                # 判别器优化
                self.encoder.eval()
                self.discriminator.zero_grad()

                z_real_gauss = Variable(torch.randn(batch_X.shape[0], self.z_dim) )

                if self.device is not 'cpu':
                    z_real_gauss = z_real_gauss.cuda()

                z_fake_gauss = self.encoder(batch_X)

                D_real_gauss, D_fake_gauss = self.discriminator(z_real_gauss), self.discriminator(z_fake_gauss)

                D_loss_gauss = -torch.mean(torch.log(D_real_gauss + self.EPS) + torch.log(1 - D_fake_gauss + self.EPS))

                D_loss_gauss.backward()
                self.optim_discriminator.step()

                self.D_loss_hist[-1] += D_loss_gauss.item()

                # 生成器优化(对抗)
                self.encoder.train()
                z_fake_gauss = self.encoder(batch_X)
                D_fake_gauss = self.discriminator(z_fake_gauss)
                G_loss = -torch.mean(torch.log(D_fake_gauss + self.EPS))
                G_loss.backward()
                self.optim_generator.step()

                self.G_loss_hist[-1] += G_loss.item()

            print('Epoch{}, 重构误差{}, 判别器判别误差{}, 生成器优化{}'.format(epoch+1, self.reconst_loss_hist[-1], self.D_loss_hist[-1], self.G_loss_hist[-1]))
        print('Optimization Finished !')
        return self

    def generate(self, sample_number=1000):

        if self.z_dim is 1:
            X = np.random.normal(loc=0.0, scale=1.0, size=sample_number)
        else:
            mean = np.zeros(self.z_dim)
            # mean = np.array([self.z_dim, 0])
            cov = np.eye(self.z_dim)
            X = np.random.multivariate_normal(mean=mean, cov=cov, size=sample_number)

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            X_generate = self.decoder(X).cpu().numpy()

        return X_generate


# 数据读取及其对应的实操
# 这里的名字改成你要的数据的名字
data = pd.read_csv('Debutanizer_Data.txt', sep = '\s+')
# 转为numpy
X_raw = np.array(data)
# lr学习率可以适当调整,设置seed随机种子是为了让结果具有重复性

Sampling = AAEModel(input_dim=data.shape[1], hidden_dim=8, z_dim=3, num_epochs=100, gen_lr=0.001, reg_lr=0.0005, EPS=1.0e-15,device='cuda:0', seed=1024,).fit(X_raw)
# sample number是进行采样的数据多少
X_generate = Sampling.generate(sample_number=1000)
# 写入csv,到时候转为excel
np.savetxt('AAE_Sampling.csv', X_generate, delimiter=',')





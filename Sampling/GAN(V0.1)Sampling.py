import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

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

# 生成器
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.Linear1 = nn.Linear(input_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.Linear3 = nn.Linear(hidden_dim, output_dim)

        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()

    def forward(self, X):

        Hidden = self.act1(self.Linear1(X))
        Hidden = self.act1(self.Linear2(Hidden))
        out = self.act2(self.Linear3(Hidden))

        return out

# 判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.Linear1 = nn.Linear(input_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.Linear3 = nn.Linear(hidden_dim, output_dim)

        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.act2 = nn.Sigmoid()

    def forward(self, X):

        Hidden = self.act1(self.Linear1(X))
        Hidden = self.act1(self.Linear2(Hidden))

        out = self.act2(self.Linear3(Hidden))

        return out

class GANModel():
    def __init__(self, input_dim, z_dim, gen_hidden_dim, dis_hidden_dim, device='cuda:0', gen_lr=0.0003, dis_lr=0.0003, num_epochs=100, batch_size=16):

        self.z_dim = z_dim
        self.input_dim = input_dim
        self.gen_hidden_dim = gen_hidden_dim
        self.dis_hidden_dim = dis_hidden_dim

        self.device = device

        self.batch_size = batch_size
        self.num_epochs = num_epochs



        self.generator = Generator(input_dim=z_dim, hidden_dim=gen_hidden_dim, output_dim=input_dim).to(device)
        self.discriminator = Discriminator(input_dim=input_dim, hidden_dim=dis_hidden_dim, output_dim=1).to(device)


        self.criterion = nn.BCELoss()

        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=gen_lr)
        self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=dis_lr)

        self.D_loss_hist = []
        self.G_loss_hist = []

        # self.real_score_hist = []
        # self.fake_score_hist = []

        self.scaler_X = StandardScaler()
        # self.scaler_X = MinMaxScaler()




    def fit(self, X):


        # X = self.scaler_X.fit_transform(X)

        dataset = MyDataset(torch.tensor(X, dtype=torch.float32, device=self.device), mode='2D')

        self.discriminator.train()
        self.generator.train()

        for epoch in range(self.num_epochs):

            self.D_loss_hist.append(0)
            self.G_loss_hist.append(0)



            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X in data_loader:

                # 真实样本的标签为1
                real_label = Variable(torch.ones((batch_X.shape[0], 1), device=self.device))
                # 虚假样本的标签为0
                fake_label = Variable(torch.zeros((batch_X.shape[0], 1), device=self.device))

                '----------------  判别器训练  ----------------'
                # 梯度清零
                self.dis_optimizer.zero_grad()

                # 真样本输出
                real_out = self.discriminator(X=batch_X)
                # 真样本的损失函数
                D_loss_real = self.criterion(real_out, real_label)



                # fake的损失
                Z = Variable(torch.randn((batch_X.shape[0], self.z_dim), device=self.device))
                batch_Fake = self.generator(Z).detach()
                fake_out = self.discriminator(batch_Fake)
                D_loss_fake = self.criterion(fake_out, fake_label)

                # 二者损失函数联合
                D_loss = D_loss_fake + D_loss_real
                D_loss.backward()
                # 参数更新
                self.dis_optimizer.step()

                self.D_loss_hist[-1] += D_loss.item()

                '----------------  生成器训练  ----------------'
                self.gen_optimizer.zero_grad()

                # 生成假样本
                Z = Variable(torch.randn((batch_X.shape[0], self.z_dim), device=self.device))
                fake_data = self.generator(Z)
                fake_data_label = self.discriminator(fake_data)
                G_loss = self.criterion(fake_data_label, real_label)

                G_loss.backward()
                self.gen_optimizer.step()

                self.G_loss_hist[-1] += G_loss.item()

            print('Epoch:{}, 判别器损失:{}, 生成器损失:{}'.format(epoch+1, self.D_loss_hist[-1], self.G_loss_hist[-1]))

        print('Optimization Finished')
        return self

    def generate(self, sample_number):

        if self.z_dim is 1:
            X = np.random.normal(loc=0.0, scale=1.0, size=sample_number)
        else:
            mean = np.zeros(self.z_dim)

            cov = np.eye(self.z_dim)
            X = np.random.multivariate_normal(mean=mean, cov=cov, size=sample_number)

        X = torch.tensor(X, dtype=torch.float32, device=self.device)

        self.generator.eval()
        with torch.no_grad():
            X_generate = self.generator(X).cpu().numpy()

        # 取决于具体情况是否要对数据做标准化
        # X_generate = self.scaler_X.inverse_transform(X_generate)

        return X_generate

# 设置随机种子
setup_seed(seed=1024)

# 数据读取及其对应的实操
# 这里的名字改成你要的数据的名字
data = pd.read_csv('Debutanizer_Data.txt', sep = '\s+')
# 转为numpy
X_raw = np.array(data)
# lr学习率可以适当调整,设置seed随机种子是为了让结果具有重复性, z_dim是GAN的隐空间的采样参数
Sampling = GANModel(input_dim=data.shape[1], z_dim=2, gen_hidden_dim=6,  dis_hidden_dim=4,  num_epochs=1000, gen_lr=0.0003,
                    dis_lr=0.0003, device='cuda:0').fit(X_raw)
# sample number是进行采样的数据多少
X_generate = Sampling.generate(sample_number=1000)
# 写入csv,到时候转为excel
np.savetxt('GAN_Sampling.csv', X_generate, delimiter=',')
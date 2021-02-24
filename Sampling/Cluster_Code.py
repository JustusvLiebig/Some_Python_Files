import os
import pandas as pd
import sklearn
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# 卡住随机种子保证结果复现性
sklearn.utils.check_random_state(1024)

# Gaussian Mixture Model 做聚类
def Data2Clustered_GMM(data, n_components):

    # 先拷贝一个数据
    frame = data.copy(deep=True)

    # 数据写入文件夹
    folder = os.getcwd()
    folder = folder + '\\GMMResults'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 获取列名
    NoColumns = data.shape[1]
    ColNameList = []

    for i in range(NoColumns):
        ColName = 'N' + str(i)
        ColNameList.append(ColName)
    ColNameList.append('Cluster')


    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(data)
    labels = gmm.predict(data)
    frame['Cluster'] = labels
    frame.columns = ColNameList

    # 分聚类写入csv文件
    for i in range(n_components):
        df_temp = frame.loc[frame['Cluster'] == i]
        Name = 'Cluster' + str(i+1)
        df_temp.to_csv(folder+'\\'+Name+'.csv', sep=',')

    return frame

# KMeans 做聚类
def Data2Clustered_KMeans(data, n_clusters):

    # 先拷贝一个数据
    frame = data.copy(deep=True)

    # 数据写入文件夹
    folder = os.getcwd()
    folder = folder + '\\KMeansResults'
    if not os.path.exists(folder):
        os.makedirs(folder)

    NoColumns = data.shape[1]
    ColNameList = []

    for i in range(NoColumns):
        ColName = 'N' + str(i)
        ColNameList.append(ColName)
    ColNameList.append('Cluster')


    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    frame['Cluster'] = labels
    frame.columns = ColNameList

    # 分聚类写入csv文件
    for i in range(n_clusters):
        df_temp = frame.loc[frame['Cluster'] == i]
        Name = 'Cluster' + str(i+1)
        df_temp.to_csv(folder+'\\'+Name+'.csv', sep=',')

    return frame

# 读取数据
data = pd.read_csv('orignaldata2.csv', sep=',', header=None, index_col=None)

after_GMM = Data2Clustered_GMM(data=data, n_components=3)
after_KMeans = Data2Clustered_KMeans(data=data, n_clusters=3)




import pandas as pd
#import numpy as np

StreamsSheet = pd.read_csv('Streams.csv', header=0, index_col=0)

def SeparateStreams(DataMatrix):
    ColdStreams_list = []
    HotStreams_list = []
    NoRows = DataMatrix.shape[0]

    Streams_index = DataMatrix._stat_axis.values

    HotStreams_Matrix = DataMatrix
    ColdStreams_Matrix = DataMatrix

    for i in range(NoRows):
        # 冷物流切割
        if DataMatrix.loc[Streams_index[i], 'Tin'] < DataMatrix.loc[Streams_index[i], 'Tout']:
            ColdStreams_list.append(Streams_index[i])
        # 热物流切割
        else:
            HotStreams_list.append(Streams_index[i])

    # 保存热物流
    for i in range(len(ColdStreams_list)):
        HotStreams_Matrix = HotStreams_Matrix.drop(ColdStreams_list[i])
    # 保存冷物流
    for i in range(len(HotStreams_list)):
        ColdStreams_Matrix = ColdStreams_Matrix.drop(HotStreams_list[i])

    return HotStreams_Matrix, ColdStreams_Matrix

HSTable, CSTable = SeparateStreams(StreamsSheet)
# 写入矩阵
HSTable.to_csv("HotProcessStreams.csv", sep=',')
CSTable.to_csv("ColdProcessStreams.csv", sep=',')

No_of_HS = HSTable.shape[0]
No_of_CS = CSTable.shape[0]
NoK = max(No_of_HS, No_of_CS)
klist1 = ['k']
klist2 = ['k']
for i in range(NoK+1):
    i = i + 1
    klist1.append('k' + str(i))
for i in range(NoK):
    i = i + 1
    klist2.append('k' + str(i))
#print(klist)
klist1_pd = pd.DataFrame(index=klist1)
klist1_pd.to_csv('Total_Stage.csv')

klist2_pd = pd.DataFrame(index=klist2)
klist2_pd.to_csv('Part_Stage.csv')



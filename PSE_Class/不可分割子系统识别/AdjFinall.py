import pandas as pd
import numpy as np

# 不可分割子系统的程序

Adj_Matrix = pd.read_csv('Adj.csv', header=0, index_col=0)

# 删掉没有输入物流的节点
def DeleteColumns(Matrix, DNE_in_label):

    # 参数预定义
    Delete_Labels = []
    ColumnLabel = Matrix.columns.values
    NoColumns = Matrix.shape[1]
    Matrix_1 = Matrix

    for i in range(NoColumns):
        # print(i)
        sum_of_ith_column = Matrix_1.iloc[:, i].sum()
        if sum_of_ith_column == 0:
            Delete_Labels.append(ColumnLabel[i])
            DNE_in_label.append(ColumnLabel[i])

    if Delete_Labels:
        for j in range(len(Delete_Labels)):
            # 删掉行
            Matrix_1 = Matrix_1.drop(Delete_Labels[j])
            # 删掉列
            Matrix_1 = Matrix_1.drop(columns=Delete_Labels[j])

    # 求列和
    column_sum_Matrix1 = Matrix_1.apply(lambda x: x.sum())
    repeat = 0
    for h in range(column_sum_Matrix1.shape[0]):
        if column_sum_Matrix1[h] == 0:
            repeat = 1
        else:
            repeat = repeat

    return Matrix_1, DNE_in_label, repeat

# 删掉没有输出物流的节点
def DeleteRows(Matrix, DNE_out_label):

    # 参数预定义
    Delete_Labels = []
    RowLabel = Matrix._stat_axis.values
    NoRows = Matrix.shape[0]
    Matrix_1 = Matrix

    for i in range(NoRows):
        # print(i)
        sum_of_ith_column = Matrix_1.iloc[i, :].sum()
        if sum_of_ith_column == 0:
            Delete_Labels.append(RowLabel[i])
            DNE_out_label.append(RowLabel[i])

    if Delete_Labels:
        for j in range(len(Delete_Labels)):
            # 删掉行
            Matrix_1 = Matrix_1.drop(Delete_Labels[j])
            # 删掉列
            Matrix_1 = Matrix_1.drop(columns=Delete_Labels[j])

    # 求列和
    row_sum_Matrix1 = Matrix_1.apply(lambda x: x.sum(), axis=1)
    repeat = 0
    for h in range(row_sum_Matrix1.shape[0]):
        if row_sum_Matrix1[h] == 0:
            repeat = 1
        else:
            repeat = repeat

    return Matrix_1, DNE_out_label, repeat

# 可及矩阵的操作
def Reach_Matrix_Multi(Matrix):
    Mul_index = Matrix.shape[0]
    Matrix_np = np.array(Matrix)
    columns_label = Matrix.columns.values
    row_label = Matrix._stat_axis.values

    Matrix_np_temp1 = Matrix_np
    Matrix_np_temp2 = Matrix_np

    for i in range(2, Mul_index):
        Matrix_np_temp1 = np.matmul(Matrix_np_temp1, Matrix_np)
        Matrix_np_temp2 = Matrix_np_temp2 + Matrix_np_temp1

    Matrix_np_R = Matrix_np_temp2
    Matrix_np_R_array = np.int64(Matrix_np_R > 0)

    Matrix_np_R_T = Matrix_np_R.transpose(1, 0)
    Matrix_np_R_T_array = np.int64(Matrix_np_R_T > 0)

    Matrix_np_R_sum = Matrix_np_R_array * Matrix_np_R_T_array
    Matrix_np_R_sum_array = np.int64(Matrix_np_R_sum > 0)

    Matrix_np_R_sum_pd = pd.DataFrame(Matrix_np_R_sum_array, index=row_label, columns=columns_label)

    return Matrix_np_R_sum_pd

# 可及矩阵的切割
def FinalCut(Matrix):
    together_list = []
    delete_list = []
    column_label = Matrix.columns.values
    row_label = Matrix._stat_axis.values
    Matrix_size = Matrix.shape[0]
    Matrix_delete = Matrix
    # 按行遍历
    for i in range(Matrix_size):
        together_list.append(row_label[i])
        delete_list.append(row_label[i])
        # 按列遍历
        for j in range(Matrix_size):
            #print(Matrix)

            #print(Matrix.iat[i, j])
            if Matrix.iat[i, j] > 0 and i != j:
                together_list.append(column_label[j])
                delete_list.append(column_label[j])
        if j == Matrix_size-1:
            break

    for i in range(len(delete_list)):
        # 删掉行
        Matrix_delete = Matrix_delete.drop(delete_list[i])
        # 删掉列
        Matrix_delete = Matrix_delete.drop(columns=delete_list[i])


    repeat = 0
    # print(Matrix_delete)

    if Matrix_delete.shape[0] > 0:
        repeat = 1

    return Matrix_delete, together_list, repeat

# 删掉没有进去的
Label = Adj_Matrix.columns.values
Adj_Matrix_Step1 = Adj_Matrix

DNE_in_label = []
repeat = 0
s = Adj_Matrix_Step1.apply(lambda x: x.sum())
for i in range(s.shape[0]):
    if s[i] == 0:
        repeat = 1
    else:
        repeat = repeat
while repeat == 1:
    Adj_Matrix_Step1, DNE_in_label, repeat = DeleteColumns(Matrix=Adj_Matrix_Step1,
                                                           DNE_in_label=DNE_in_label)


# 删掉没有出来的
Adj_Matrix_Step2 = Adj_Matrix_Step1
DNE_out_label = []
repeat = 0
s = Adj_Matrix_Step1.apply(lambda x: x.sum(), axis=1)
for i in range(s.shape[0]):
    if s[i] == 0:
        repeat = 1
    else:
        repeat = repeat
while repeat == 1:
    Adj_Matrix_Step2, DNE_out_label, repeat = DeleteRows(Matrix=Adj_Matrix_Step2,
                                                         DNE_out_label=DNE_out_label)

# 打印结果
print('以下节点为没有输入物流的节点,分别为不可分割子系统:')
print(DNE_in_label)
print('以下节点为没有输出物流的节点,分别为不可分割子系统:')
print(DNE_out_label)

# 进行最后的系统切割
Adj_Matrix_Step3 = Reach_Matrix_Multi(Adj_Matrix_Step2)

repeat = 0

Adj_Matrix_Step4 = Adj_Matrix_Step3
repeat = 1
while repeat == 1:
    Adj_Matrix_Step4, system_list, repeat = FinalCut(Matrix=Adj_Matrix_Step4)
    print('以下节点为存在回路连接的节点,总体属于不可分割子系统:')
    print(system_list)












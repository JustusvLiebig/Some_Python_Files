import networkx as nx
import sys
import pyomo.environ as pe
import pandas as pd


def get_loops(Matrix):
    ColumnLabels = Matrix.columns.values
    RowLabels = Matrix._stat_axis.values

    G = nx.DiGraph()

    G.add_nodes_from(RowLabels)

    reach_list = []
    stream_name = []
    node_list = []

    h = 1
    for i in range(len(RowLabels)):
        for j in range(len(ColumnLabels)):
            if Matrix.iloc[i, j] == 1:
                reach_list.append((str(RowLabels[i]), ColumnLabels[j]))
                stream_id = 'S' + str(h)
                stream_name.append(stream_id)
                node_list.append([str(RowLabels[i]), ColumnLabels[j]])
                # stream_list.append([stream_id,str(RowLabels[i]), ColumnLabels[j]])
                h = h + 1

    G.add_edges_from(reach_list)
    # print(stream_name)
    # print(node_list)

    # stream_node是节点和对应的物流名称矩阵(给节点之间连接物流命名)
    stream_node = pd.DataFrame(node_list, index=stream_name, columns=['output_node', 'input_node'])
    #print('节点和其对应的物流的命名矩阵是:')
    #print(stream_node)


    loop_list = list(nx.simple_cycles(G))
    list_null = []
    if loop_list == list_null:
        print('没有回路程序不用执行！')
        sys.exit(0)

    # loop_list是回路包含的节点列表
    print('有{}个回路,'.format(len(loop_list)))
    print('分别包含的节点是:')

    loop_name = []


    for i in range(len(loop_list)):
        loop_id = 'L' + str(i + 1)
        loop_name.append(loop_id)
        print(loop_list[i])


    # loop_stream是环-物流的之间的关系矩阵，Loop Matrix
    loop_stream = pd.DataFrame(index=stream_name, columns=loop_name)
    for i in range(len(loop_list)):
        for j in range(len(stream_name)):
            judge_list = stream_node.iloc[j, :].tolist()

            # 判断路径是不是在环里
            subset_judge = set(judge_list) <= set(loop_list[i])
            if subset_judge == True:
                loop_stream.iloc[j, i] = 1
            elif subset_judge == False:
                loop_stream.iloc[j, i] = 0

    loop_stream = loop_stream.T

    return stream_node, loop_stream, loop_name, stream_name



Loop_Matrix = pd.read_csv('Loop2.csv', header=0, index_col=0)
stream_node, loop_stream, loop_name, stream_name = get_loops(Loop_Matrix)
# 物流的命名矩阵
stream_node.to_csv('物流命名.csv', sep=',')
#loop_stream = loop_stream.T
# 回路矩阵写出
loop_stream.to_csv('Loop_Matrix.csv', sep=',')



model = pe.ConcreteModel()
#model.name(['循环物流断裂的混合整数规划问题'])

model.Streams = pe.Set(initialize=stream_name)
model.Loops = pe.Set(initialize=loop_name)
model.X = pe.Var(model.Streams, domain=pe.Binary)
#print(loop_stream)
Big_C_init = loop_stream.stack().to_dict()
#print(Big_C_init)
model.BigC = pe.Param(model.Loops, model.Streams, initialize=Big_C_init)
#model.BigC = pe.Param(initialize=loop_stream.T)

#TearMIP = model.create_instance(data)
#data.load(filename='Stream_Loop.csv', param=model.BigC,  format='array')

model.Tear1 = pe.ConstraintList()
for i in model.Loops:
    model.Tear1.add(sum((model.X[j] * model.BigC[i, j]) for j in model.Streams) >= 1)

model.obj = pe.Objective(expr=(sum(model.X[j] for j in model.Streams)), sense=pe.minimize)
solver = pe.SolverFactory('gurobi')
solver.options['threads'] = 4
print('-------------------进行模型求解-------------------')
results = solver.solve(model)
print('-------------------求解器状态展示-------------------')
print("Gurobi的求解状态是:"+str(results.solver.status))
print('要想把这几个节点全部解开需要断裂如下的物流:')

#print('物流节点对应关系是:')
#print(stream_node)
#print(stream_node.loc['S1', 'output_node'])
for v in model.component_objects(pe.Var, active=True):
    if v is model.X:
        for index in v:
            if pe.value(v[index]) >= 0.5:
                #print('应该断裂的物流是:', index)
                print(stream_node.loc[index, 'output_node'], '------>', stream_node.loc[index, 'input_node'])
import pandas as pd
import pyomo
from pyomo.environ import *



model = AbstractModel()
data = DataPortal()

model.HS = Set()
model.thin = Param(model.HS)
model.thout = Param(model.HS)
model.fh = Param(model.HS)
model.hh = Param(model.HS)
model.EHS = Param(model.HS, mutable=True)


model.CS = Set()
model.tcin = Param(model.CS)
model.tcout = Param(model.CS)
model.fc = Param(model.CS)
model.hc = Param(model.CS)
model.ECS = Param(model.CS, mutable=True)

model.gamma = Param(model.HS, model.CS, mutable=True)

model.thuin = Param(mutable=True, initialize=350)
model.thuout = Param(mutable=True, initialize=345)
model.tcuin = Param(mutable=True, initialize=10)
model.tcuout = Param(mutable=True, initialize=15)
model.tmapp = Param(initialize=10)

model.unitc = Param(initialize=500)
model.hucost = Param(initialize=200)
model.cucost = Param(initialize=10)

model.acoeff = Param(initialize=146)
model.hucoeff = Param(initialize=146)
model.cucoeff = Param(initialize=146)
model.hhu = Param(initialize=3.4)
model.hcu = Param(initialize=1.7)


model.k = Set(ordered=True)
model.st = Set(ordered=True)

data.load(filename='HotProcessStreams.csv', index=model.HS, param=(model.thin, model.thout, model.fh, model.hh))
data.load(filename='ColdProcessStreams.csv', index=model.CS, param=(model.tcin, model.tcout, model.fc, model.hc))
data.load(filename='Total_Stage.csv', set=model.k)
data.load(filename='Part_Stage.csv', set=model.st)



model.th = Var(model.HS, model.k, domain=PositiveReals)
model.tc = Var(model.CS, model.k, domain=PositiveReals)
model.q = Var(model.HS, model.CS, model.st, domain=NonNegativeReals)
model.qh = Var(model.CS, domain=NonNegativeReals)
model.qc = Var(model.HS, domain=NonNegativeReals)
model.dt = Var(model.HS, model.CS, model.k, domain=PositiveReals)
model.dthu = Var(model.CS, domain=PositiveReals)
model.dtcu = Var(model.HS, domain=PositiveReals)

model.ChenTD = Var(model.HS, model.CS, model.st, domain=NonNegativeReals)
model.ChenTDhu = Var(model.CS, domain=NonNegativeReals)
model.ChenTDcu = Var(model.HS, domain=NonNegativeReals)



#model.z = Var(model.HS, model.CS, model.k, bounds=(0, 1.5), initialize=0, within=PositiveIntegers)
#model.zcu = Var(model.HS, bounds=(0, 1.5), initialize=0, within=PositiveIntegers)
#model.zhu = Var(model.CS, bounds=(0, 1.5), initialize=0, within=PositiveIntegers)


model.z = Var(model.HS, model.CS, model.st, initialize=0, within=Binary)
model.zcu = Var(model.HS, initialize=0, within=Binary)
model.zhu = Var(model.CS, initialize=0, within=Binary)

# model.cost = Objective


# 实例化模型
SYNHEAT = model.create_instance(data)

#SYNHEAT.name('Stage-wise Superstructure for Heat Exchanger Network Synthesis')

thuin = value(SYNHEAT.thuin)
tcuin = value(SYNHEAT.tcuin)

'''

SYNHEAT.z = Var(SYNHEAT.HS, SYNHEAT.CS, SYNHEAT.k, initialize=0, within=Binary)
SYNHEAT.zcu = Var(SYNHEAT.HS, initialize=0, within=Binary)
SYNHEAT.zhu = Var(SYNHEAT.CS, initialize=0, within=Binary)
'''

# 全局热量守恒
SYNHEAT.teh = ConstraintList()
SYNHEAT.tec = ConstraintList()
for i in SYNHEAT.HS:
    SYNHEAT.teh.add(sum(sum(SYNHEAT.q[i, j, k] for j in SYNHEAT.CS) for k in SYNHEAT.st) + SYNHEAT.qc[i] == SYNHEAT.fh[i] * (SYNHEAT.thin[i] - SYNHEAT.thout[i]))
# teh(i).. (thin(i)-thout(i))*fh(i) =e= sum((j,st), q(i,j,st)) + qc(i) ;
for j in SYNHEAT.CS:
    SYNHEAT.tec.add(sum(sum(SYNHEAT.q[i, j, k] for i in SYNHEAT.HS) for k in SYNHEAT.st) + SYNHEAT.qh[j] == SYNHEAT.fc[j] * (SYNHEAT.tcout[j] - SYNHEAT.tcin[j]))
# tec(j).. (tcout(j)-tcin(j))*fc(j) =e= sum((i,st), q(i,j,st)) + qh(j) ;

# 局部热量守恒
SYNHEAT.eh = ConstraintList()
SYNHEAT.ec = ConstraintList()

for i in SYNHEAT.HS:
    for k in SYNHEAT.k:
        if k is not SYNHEAT.k.last():
            SYNHEAT.eh.add(SYNHEAT.fh[i] * (SYNHEAT.th[i, k] - SYNHEAT.th[i, SYNHEAT.k.next(k)]) == sum(SYNHEAT.q[i, j, k] for j in SYNHEAT.CS))
# eh(i,k)$st(k).. fh(i)*(th(i,k) - th(i,k+1)) =e= sum(j, q(i,j,k))

for j in SYNHEAT.CS:
    for k in SYNHEAT.k:
        if k is not SYNHEAT.k.last():
            SYNHEAT.ec.add(SYNHEAT.fc[j] * (SYNHEAT.tc[j, k] - SYNHEAT.tc[j, SYNHEAT.k.next(k)]) == sum(SYNHEAT.q[i, j, k] for i in SYNHEAT.HS))
# ec(j,k)$st(k).. fc(j)*(tc(j,k) - tc(j,k+1)) =e= sum(i,q(i,j,k))

# 公用工程用量计算
SYNHEAT.eqh = ConstraintList()
SYNHEAT.eqc = ConstraintList()

for j in SYNHEAT.CS:
    for k in SYNHEAT.k:
        if k is SYNHEAT.k.first():
            SYNHEAT.eqh.add(SYNHEAT.fc[j] * (SYNHEAT.tcout[j] - SYNHEAT.tc[j, k]) == SYNHEAT.qh[j])
# eqh(j,k)$first(k).. fc(j)*(tcout(j) - tc(j,k)) =e= qh(j) ;

for i in SYNHEAT.HS:
    for k in SYNHEAT.k:
        if k is SYNHEAT.k.last():
            SYNHEAT.eqc.add(SYNHEAT.fh[i] * (SYNHEAT.th[i, k] - SYNHEAT.thout[i]) == SYNHEAT.qc[i])
# eqc(i,k)$last(k)..  fh(i)*(th(i,k) - thout(i)) =e= qc(i) ;

# 进出口温度赋值
SYNHEAT.tinh = ConstraintList()
SYNHEAT.tinc = ConstraintList()

for i in SYNHEAT.HS:
    for k in SYNHEAT.k:
        if k is SYNHEAT.k.first():
            SYNHEAT.tinh.add(SYNHEAT.thin[i] == SYNHEAT.th[i, k])
# tinh(i,k)$first(k).. thin(i) =e= th(i,k) ;

for j in SYNHEAT.CS:
    for k in SYNHEAT.k:
        if k is SYNHEAT.k.last():
            SYNHEAT.tinc.add(SYNHEAT.tcin[j] == SYNHEAT.tc[j, k])
#  tinc(j,k)$last(k)..  tcin(j) =e= tc(j,k) ;

# 单调性给定
SYNHEAT.month = ConstraintList()
SYNHEAT.montc = ConstraintList()

for i in SYNHEAT.HS:
    for k in SYNHEAT.k:
        if k is not SYNHEAT.k.last():
            SYNHEAT.month.add(SYNHEAT.th[i, k] >= SYNHEAT.th[i, SYNHEAT.k.next(k)])

# month(i,k)$st(k).. th(i,k) =g= th(i,k+1)


for j in SYNHEAT.CS:
    for k in SYNHEAT.k:
        if k is not SYNHEAT.k.last():
            SYNHEAT.montc.add(SYNHEAT.tc[j, k] >= SYNHEAT.tc[j, SYNHEAT.k.next(k)])
# montc(j,k)$st(k).. tc(j,k) =g= tc(j,k+1)
# 进出口单调性
SYNHEAT.monthl = ConstraintList()
SYNHEAT.montcf = ConstraintList()

for i in SYNHEAT.HS:
    for k in SYNHEAT.k:
        if k is SYNHEAT.k.last():
            SYNHEAT.monthl.add(SYNHEAT.th[i, k] >= SYNHEAT.thout[i])
# monthl(i,k)$last(k).. th(i,k) =g= thout(i) ;

for j in SYNHEAT.CS:
    for k in SYNHEAT.k:
        if k is SYNHEAT.k.first():
            SYNHEAT.montcf.add(SYNHEAT.tcout[j] >= SYNHEAT.tc[j, k])
# montcf(j,k)$first(k)..tcout(j) =g= tc(j,k) ;

# 换热量的逻辑约束
# 最大换热量参数计算
for i in SYNHEAT.HS:
    SYNHEAT.EHS[i] = SYNHEAT.fh[i] * (SYNHEAT.thin[i] - SYNHEAT.thout[i])
for j in SYNHEAT.CS:
    SYNHEAT.ECS[j] = SYNHEAT.fc[j] * (SYNHEAT.tcout[j] - SYNHEAT.tcin[j])

# 换热量逻辑约束
SYNHEAT.logq = ConstraintList()
SYNHEAT.logqc = ConstraintList()
SYNHEAT.logqh = ConstraintList()
for i in SYNHEAT.HS:
    for j in SYNHEAT.CS:
        for k in SYNHEAT.k:
            if k is not SYNHEAT.k.last():
                SYNHEAT.logq.add(SYNHEAT.q[i, j, k] <= min(value(SYNHEAT.EHS[i]), value(SYNHEAT.ECS[j])) * SYNHEAT.z[i, j, k])
                # q(i, j, k) - min(ech(i), ecc(j)) * z(i, j, k) = l = 0
                # SYNHEAT.logq.add(SYNHEAT.q[i, j, k] >= 0)
                SYNHEAT.logq.add(SYNHEAT.z[i, j, k] >= 0)
                SYNHEAT.logq.add(SYNHEAT.z[i, j, k] <= 1)

for i in SYNHEAT.HS:
    SYNHEAT.logqc.add(SYNHEAT.qc[i] <= value(SYNHEAT.EHS[i]) * SYNHEAT.zcu[i])
    # qc(i) - ech(i)*zcu(i) =l= 0 ;
    # SYNHEAT.logqc.add(SYNHEAT.qc[i] >= 0)
    SYNHEAT.logq.add(SYNHEAT.zcu[i] >= 0)
    SYNHEAT.logq.add(SYNHEAT.zcu[i] <= 1)

for j in SYNHEAT.CS:
    SYNHEAT.logqh.add(SYNHEAT.qh[j] <= value(SYNHEAT.ECS[j]) * SYNHEAT.zhu[j])
    # qh(j) - ecc(j) * zhu(j) = l = 0;
    # SYNHEAT.logqh.add(SYNHEAT.qh[j] >= 0)

# 换热温差逻辑约束
SYNHEAT.logdth = ConstraintList()
SYNHEAT.logdtc = ConstraintList()
SYNHEAT.logdthu = ConstraintList()
SYNHEAT.logdtcu = ConstraintList()
for i in SYNHEAT.HS:
    for j in SYNHEAT.CS:
        SYNHEAT.gamma[i, j] = max(0,
                                  value(SYNHEAT.tcin[j]) - value(SYNHEAT.thout[i]),
                                  value(SYNHEAT.tcin[j]) - value(SYNHEAT.thin[i]),
                                  value(SYNHEAT.tcout[j]) - value(SYNHEAT.thin[i]),
                                  value(SYNHEAT.tcout[j]) - value(SYNHEAT.thout[i]))

for i in SYNHEAT.HS:
    for j in SYNHEAT.CS:
        for k in SYNHEAT.k:
            if k is not SYNHEAT.k.last():
                SYNHEAT.logdth.add(SYNHEAT.dt[i, j, k] <= SYNHEAT.th[i, k]
                                   - SYNHEAT.tc[j, k]
                                   + SYNHEAT.gamma[i, j] * (1 - SYNHEAT.z[i, j, k]))
                SYNHEAT.logdtc.add(SYNHEAT.dt[i, j, SYNHEAT.k.next(k)] <= SYNHEAT.th[i, SYNHEAT.k.next(k)]
                                   - SYNHEAT.tc[j, SYNHEAT.k.next(k)]
                                   + SYNHEAT.gamma[i, j] * (1 - SYNHEAT.z[i, j, k]))

for j in SYNHEAT.CS:
    for k in SYNHEAT.k:
        if k is SYNHEAT.k.first():
            SYNHEAT.logdthu.add(SYNHEAT.dthu[j] <= SYNHEAT.thuout - SYNHEAT.tc[j, k])

for i in SYNHEAT.HS:
    for k in SYNHEAT.k:
        if k is SYNHEAT.k.last():
            SYNHEAT.logdtcu.add(SYNHEAT.dtcu[i] <= SYNHEAT.th[i, k] - SYNHEAT.tcuout)

# 边界约束
SYNHEAT.mindt = ConstraintList()
SYNHEAT.mindtcu = ConstraintList()
SYNHEAT.mindthu = ConstraintList()
SYNHEAT.boundsth = ConstraintList()
SYNHEAT.boundstc = ConstraintList()
for i in SYNHEAT.HS:
    for j in SYNHEAT.CS:
        for k in SYNHEAT.k:
            SYNHEAT.mindt.add(SYNHEAT.dt[i, j, k] >= SYNHEAT.tmapp)

for j in SYNHEAT.CS:
    SYNHEAT.mindtcu.add(SYNHEAT.dthu[j] >= SYNHEAT.tmapp)

for i in SYNHEAT.HS:
    SYNHEAT.mindthu.add(SYNHEAT.dtcu[i] >= SYNHEAT.tmapp)

for i in SYNHEAT.HS:
    for k in SYNHEAT.k:
        SYNHEAT.boundsth.add(SYNHEAT.th[i, k] <= value(SYNHEAT.thin[i]))
        SYNHEAT.boundsth.add(SYNHEAT.th[i, k] >= value(SYNHEAT.thout[i]))

for j in SYNHEAT.CS:
    for k in SYNHEAT.k:
        SYNHEAT.boundstc.add(SYNHEAT.tc[j, k] <= value(SYNHEAT.tcout[j]))
        SYNHEAT.boundstc.add(SYNHEAT.tc[j, k] >= value(SYNHEAT.tcin[j]))

# objective
#revenue = sum(sum(m.x[s,p]*products[p]['price'] for s in S) for p in P)
#cost = sum(sum(m.x[s,p]*streams[s]['cost'] for s in S) for p in P)
#m.profit = pyomo.Objective(expr = revenue - cost, sense=pyomo.maximize)





# 温差计算
SYNHEAT.HEXdt = ConstraintList()
SYNHEAT.HHUdt = ConstraintList()
SYNHEAT.HCUdt = ConstraintList()
for i in SYNHEAT.HS:
    for j in SYNHEAT.CS:
        for k in SYNHEAT.k:
            if k is not SYNHEAT.k.last():
                '''
                SYNHEAT.HEXdt.add(SYNHEAT.ChenTD[i, j, k] == (SYNHEAT.dt[i, j, k]
                                                              * SYNHEAT.dt[i, j, SYNHEAT.k.next(k)]
                                                              * (SYNHEAT.dt[i, j, k] + SYNHEAT.dt[i, j, SYNHEAT.k.next(k)])/2
                                                              + 1.0E-10) ** (1/3) + 1.0E-10)
                '''
                SYNHEAT.HEXdt.add(SYNHEAT.ChenTD[i, j, k] == 0.5 * (SYNHEAT.dt[i, j, k] + SYNHEAT.dt[i, j, SYNHEAT.k.next(k)]))
                #SYNHEAT.HEXdt.add(SYNHEAT.ChenTD[i, j, k] >= SYNHEAT.tmapp)

for j in SYNHEAT.CS:
    '''
    
    SYNHEAT.HHUdt.add(SYNHEAT.ChenTDhu[j] == ((value(SYNHEAT.thuin) - SYNHEAT.tcout[j])
                                              * SYNHEAT.dthu[j]
                                              * (value(SYNHEAT.thuin)-SYNHEAT.tcout[j]+SYNHEAT.dthu[j]) / 2
                                              + 1.0E-10) ** (1/3) + 1.0E-10)
    '''
    SYNHEAT.HHUdt.add(SYNHEAT.ChenTDhu[j] == 0.5 * (value(SYNHEAT.thuin) + SYNHEAT.tcout[j]))
    #SYNHEAT.HEXdt.add(SYNHEAT.ChenTDhu[j] >= SYNHEAT.tmapp)

for i in SYNHEAT.HS:
    '''
    
    SYNHEAT.HCUdt.add(SYNHEAT.ChenTDcu[i] == ((SYNHEAT.thout[i] - value(SYNHEAT.tcuin))
                                              * SYNHEAT.dtcu[i]
                                              * (SYNHEAT.thout[i] - value(SYNHEAT.tcuin) + SYNHEAT.dtcu[i]) / 2
                                              + 1.0E-4) ** (1/3) + 1.0E-4)
    '''
    SYNHEAT.HCUdt.add(SYNHEAT.ChenTDcu[i] == 0.5 * (SYNHEAT.thout[i] + value(SYNHEAT.tcuin)))
    #SYNHEAT.HEXdt.add(SYNHEAT.ChenTDcu[i] >= SYNHEAT.tmapp)


# 目标函数费用
install_cost = SYNHEAT.unitc * (sum(sum(sum(SYNHEAT.z[i, j, k] for i in SYNHEAT.HS) for j in SYNHEAT.CS) for k in SYNHEAT.st)
                                + sum(SYNHEAT.zcu[i] for i in SYNHEAT.HS) + sum(SYNHEAT.zhu[j] for j in SYNHEAT.CS))

utility_cost = sum((SYNHEAT.hucost * SYNHEAT.qh[j]) for j in SYNHEAT.CS) + sum((SYNHEAT.cucost * SYNHEAT.qc[i]) for i in SYNHEAT.HS)

AHEX_cost = SYNHEAT.acoeff * sum(sum(sum((SYNHEAT.q[i, j, k]
                                          * (1/SYNHEAT.hh[i] + 1/SYNHEAT.hc[j])
                                          / (SYNHEAT.ChenTD[i, j, k]))for i in SYNHEAT.HS)
                                     for j in SYNHEAT.CS)
                                 for k in SYNHEAT.st)

AHU_cost = SYNHEAT.hucoeff * (sum((SYNHEAT.qh[j] * (1/SYNHEAT.hc[j] + value(1/SYNHEAT.hhu)) / (SYNHEAT.ChenTDhu[j]))
                                  for j in SYNHEAT.CS))

ACU_cost = SYNHEAT.cucoeff * (sum((SYNHEAT.qc[i] * (1/SYNHEAT.hh[i] + value(1/SYNHEAT.hcu)) / (SYNHEAT.ChenTDcu[i]))
                                  for i in SYNHEAT.HS))

''''''
SYNHEAT.Total_cost = Objective(expr=(
    (SYNHEAT.unitc * (sum(sum(sum(SYNHEAT.z[i, j, k] for i in SYNHEAT.HS) for j in SYNHEAT.CS) for k in SYNHEAT.st)
                      + sum(SYNHEAT.zcu[i] for i in SYNHEAT.HS) + sum(SYNHEAT.zhu[j] for j in SYNHEAT.CS)))
    + (sum((SYNHEAT.hucost * SYNHEAT.qh[j]) for j in SYNHEAT.CS)
       + sum((SYNHEAT.cucost * SYNHEAT.qc[i]) for i in SYNHEAT.HS))
    + (SYNHEAT.acoeff * sum(sum(sum((SYNHEAT.q[i, j, k]
                                     * (1/SYNHEAT.hh[i] + 1/SYNHEAT.hc[j])
                                     / (SYNHEAT.ChenTD[i, j, k]))for i in SYNHEAT.HS)
                                for j in SYNHEAT.CS)
                            for k in SYNHEAT.st))
    + (SYNHEAT.hucoeff * (sum((SYNHEAT.qh[j] * (1/SYNHEAT.hc[j] + value(1/SYNHEAT.hhu)) / (SYNHEAT.ChenTDhu[j]))
                              for j in SYNHEAT.CS)))
    + (SYNHEAT.cucoeff * (sum((SYNHEAT.qc[i] * (1/SYNHEAT.hh[i] + value(1/SYNHEAT.hcu)) / (SYNHEAT.ChenTDcu[i]))
                              for i in SYNHEAT.HS)))), sense=minimize)


#SYNHEAT.Total_cost = Objective(expr=install_cost + utility_cost, sense=minimize)
#SYNHEAT.pprint()


#SolverFactory('mindtpy').option["mipgap"] = 0.1
solver = SolverFactory('mindtpy')
# solver.options['tol'] = 1E-5
results = solver.solve(SYNHEAT, mip_solver='gurobi', nlp_solver='ipopt', time_limit=3600,
                       tee=True, strategy='OA', bound_tolerance=1.0E-9, iteration_limit=100,
                       threads=6)
#SYNHEAT.display()
#SolverFactory('gdpopt').solve(SYNHEAT, mip_solver='gurobi', nlp_solver='ipopt', tee=True, strategy='LOA')

print('-------------------求解状态展示-------------------')
print("MindtPy的求解状态是:"+str(results.solver.status))

print('-------------------求解结果展示-------------------')
for v in SYNHEAT.component_objects(Var, active=True):
    # 展示换热器匹配关系
    if v is SYNHEAT.z:
        for index in v:
            if value(v[index]) >= 0.5:
                print('换热器之间的匹配关系:', index)
                print('对应的换热负荷大小是:{}'.format(value(SYNHEAT.q[index]), '.6f'), 'kW')
                print('对应的换热面积大小是:{}'.format((value(SYNHEAT.q[index]) * (
                            1 / SYNHEAT.hh[index[0]] + 1 / SYNHEAT.hc[index[1]]) / value(SYNHEAT.ChenTD[index])), '.6f'), 'm2')

    elif v is SYNHEAT.zhu:
        for index in v:
            if value(v[index]) >= 0.5:
                print('与热公用工程换热器之间的匹配关系:', index)
                print('对应的热公用工程负荷大小是:{}'.format(value(SYNHEAT.qh[index]), '.6f'), 'kW')
                print('对应的热公用工程换热面积大小是:{}'.format((value(SYNHEAT.qh[index]) * (1 / SYNHEAT.hc[index] + value(1 / SYNHEAT.hhu)) / value(SYNHEAT.ChenTDhu[index])), '.6f'), 'm2')

    elif v is SYNHEAT.zcu:
        for index in v:
            if value(v[index]) >= 0.5:
                print('与冷公用工程换热器之间的匹配关系:', index)
                print('对应的冷公用工程负荷大小是:{}'.format(value(SYNHEAT.qc[index]), '.6f'), 'kW')
                print('对应的冷公用工程换热面积大小是:{}'.format((value(SYNHEAT.qc[index]) * (1 / SYNHEAT.hh[index] + value(1 / SYNHEAT.hcu)) / value(SYNHEAT.ChenTDcu[index])), '.6f'), 'm2')

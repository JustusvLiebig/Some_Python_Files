import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import math

input_table = pd.read_csv('Streams.csv', header=0)
#print(input_table)
input_table['Heat Load (kW)'] = abs((input_table.loc[:, 'Tin'] - input_table.loc[:, 'Tout']) * input_table.loc[:, 'Fcp'])
input_table = input_table.loc[:, ['Unnamed: 0', 'Tin', 'Tout', 'Heat Load (kW)']]
input_table.columns = ['Stream Information', 'Supply Temperture (°C)', 'Target Temperature (°C)', 'Heat Load (kW)']
input_table["Stream Type"]=np.where(input_table["Supply Temperture (°C)"]>input_table["Target Temperature (°C)"],"HOT","COLD")
input_table["Heat Capacity Flowrate (kW/K)"]=round(input_table["Heat Load (kW)"]/(input_table["Target Temperature (°C)"]-input_table["Supply Temperture (°C)"]),2)
#Heat Capacity Flowrate values of all hot streams have been made negative. This convention will be used later to simplify calculations for problem table algorithm.
index=[]
for n in range(1,len(input_table)+1):
    index.append(n)
input_table["Stream Number"]=index
input_table=input_table.set_index('Stream Number')
input_table


input_table=input_table.rename(columns={"Supply Temperture (°C)": "Ts", "Target Temperature (°C)": "Tt", "Heat Capacity Flowrate (kW/K)":"FCp"})

hot_streams=input_table[input_table["Stream Type"]=="HOT"]
cold_streams=input_table[input_table["Stream Type"]=="COLD"]


Tmin=float(input("Enter your ΔTmin (°C): "))


temp_left=cold_streams["Ts"].append(cold_streams["Tt"]).append(round((hot_streams["Ts"]-Tmin),2)).append(round((hot_streams["Tt"]-Tmin),2)).sort_values(ascending=False)
temp_left=list(dict.fromkeys(temp_left)) #this will drop duplicates
temp_right=hot_streams["Ts"].append(hot_streams["Tt"]).append(round((cold_streams["Ts"]+Tmin),2)).append(round((cold_streams["Tt"]+Tmin),2)).sort_values(ascending=False)
temp_right=list(dict.fromkeys(temp_right)) #this will drop duplicates
problem_table=pd.DataFrame()
problem_table["Network"]=[i for i in range(len(temp_left))]
problem_table["Network"]=problem_table["Network"].astype(str)
problem_table["Sub"]=np.repeat("SN",len(temp_left))
problem_table['Subnetwork'] = problem_table['Sub']+problem_table["Network"]
problem_table=problem_table.drop(["Sub","Network"],axis=1)
problem_table["Tleft"]=temp_left
problem_table["Tright"]=temp_right

def Temperature_Interval_Diagram():
    plt.figure(figsize=(20,20))
    plt.xlim(0,len(input_table.index)+1)
    plt.ylim(min(0.9*problem_table["Tleft"].min(),1.1*problem_table["Tleft"].min()),max(1.1*problem_table["Tleft"].max(),0.9*problem_table["Tleft"].max()))
    for stream_number,Ts,Tt,stream_name in zip(range(len(cold_streams)),cold_streams["Ts"],cold_streams["Tt"],cold_streams.index):
        plt.arrow(stream_number+1,Ts,0,Tt-Ts,length_includes_head=True,head_length=1.0,head_width=0.1)
        plt.axhline(Ts,color='black',lw=0.5,linestyle='dashed')
        plt.axhline(Tt,color='black',lw=0.5,linestyle='dashed')
        plt.text(stream_number+1,min(0.9*problem_table["Tleft"].min(),1.1*problem_table["Tleft"].min()),stream_name,fontsize=15)
    for stream_number,Ts,Tt,stream_name in zip(range(len(cold_streams),len(cold_streams)+len(hot_streams)),hot_streams["Ts"],hot_streams["Tt"],hot_streams.index):
        plt.arrow(stream_number+1,Ts-Tmin,0,Tt-Ts,length_includes_head=True,head_length=1.0,head_width=0.1)
        plt.axhline(Ts-Tmin,color='black',lw=0.5,linestyle='dashed')
        plt.axhline(Tt-Tmin,color='black',lw=0.5,linestyle='dashed')
        plt.text(stream_number+1,min(0.9*problem_table["Tleft"].min(),1.1*problem_table["Tleft"].min()),stream_name,fontsize=15)
    plt.axvline(len(cold_streams)+0.5,color='red',lw=0.5)
    for j in problem_table["Tleft"]:
        plt.text(len(cold_streams)+0.5,j,j,ha='right')
        plt.text(len(cold_streams)+0.5,j,j+Tmin)
    plt.xticks([])
    plt.yticks([])
    plt.title('Temperature Interval Diagram',fontsize=40)
    plt.savefig('Temperature Interval Diagram.png', dpi=300, bbox_inches='tight')
    plt.show()


streams_involved=np.zeros((len(problem_table),len(input_table)+1),dtype='int')
cold_streams_involved_1=np.zeros((len(problem_table),len(input_table)+1),dtype='int')
hot_streams_involved_1=np.zeros((len(problem_table),len(input_table)+1),dtype='int')
for i in range(len(problem_table)):
    for j in range(len(cold_streams)):
        cold_streams_involved=np.where(cold_streams.iloc[j]["Ts"]<=problem_table.iloc[i]["Tleft"]<cold_streams.iloc[j]["Tt"],cold_streams.iloc[j].name,0)
        streams_involved[i][cold_streams_involved]=cold_streams_involved
        cold_streams_involved_1[i][cold_streams_involved]=cold_streams_involved
    for k in range(len(hot_streams)):
        hot_streams_involved=np.where(hot_streams.iloc[k]["Tt"]<=problem_table.iloc[i]["Tright"]<hot_streams.iloc[k]["Ts"],hot_streams.iloc[k].name,0)
        streams_involved[i][hot_streams_involved]=hot_streams_involved
        hot_streams_involved_1[i][hot_streams_involved]=hot_streams_involved
problem_table["Streams Involved"]=[streams_involved[i] for i in range(len(problem_table))]
difference=[0]
for i in range(1,len(problem_table)):
    difference.append(problem_table["Tleft"][i-1]-problem_table["Tleft"][i])
problem_table["Temperature Difference"]=difference
k=np.zeros(len(problem_table))
for i in range(len(problem_table)):
    streams=np.extract(problem_table["Streams Involved"][i]!=0, problem_table["Streams Involved"][i])
    sum_1=0#this is done so that there isn't error in the future
    n=len(streams)
    for n in range(len(streams)):
        sum_1=sum_1+input_table.loc[streams[n]]["FCp"]
    k[i]=sum_1
problem_table["Summation of FCp"]=k
problem_table["Deficit"]=problem_table["Summation of FCp"]*problem_table["Temperature Difference"]
x=[-problem_table["Deficit"].cumsum()[i-1] for i in range(1,len(problem_table))]
x=np.array(x)
y=[0]
y=np.array(y)
z=np.concatenate((y,x))
problem_table["Accumulated Input"]=z
problem_table["Accumulated Output"]=-problem_table["Deficit"].cumsum()
problem_table["Heat Flows Input"]=abs(problem_table["Accumulated Output"].min())+problem_table["Accumulated Input"]
problem_table["Heat Flows Output"]=abs(problem_table["Accumulated Output"].min())+problem_table["Accumulated Output"]


pinch_temp=problem_table[problem_table["Heat Flows Output"]==0.0]["Tright"]
pinch_temp=pinch_temp.values
pinch_temp=pinch_temp[0]
minimum_hot_utility=problem_table.iloc[0]["Heat Flows Input"]
minimum_cold_utility=problem_table.iloc[-1]["Heat Flows Output"]


hot_composite_curve=pd.DataFrame()
hot_composite_curve["Temperature"]=problem_table["Tright"].values[::-1]
hot_composite_curve["Streams Involved"]=[hot_streams_involved_1[len(hot_streams_involved_1)-i-1] for i in range(len(hot_streams_involved_1))]
k=np.zeros(len(problem_table))
for i in range(len(problem_table)):
    streams=np.extract(hot_composite_curve["Streams Involved"][i]!=0, hot_composite_curve["Streams Involved"][i])
    sum_2=0
    n=len(streams)
    for n in range(len(streams)):
        sum_2=sum_2+input_table.loc[streams[n]]["FCp"]
    k[i]=sum_2
hot_composite_curve["Summation of FCp"]=k
hot_composite_curve["Difference"]=problem_table["Temperature Difference"].values[::-1]
hot_composite_curve["Enthalpy not Final"]=hot_composite_curve["Difference"]*hot_composite_curve["Summation of FCp"]
hot_composite_curve["Enthalpy not Final Cumulative"]=hot_composite_curve["Enthalpy not Final"].cumsum()
final_enthalpy_hot_streams=np.concatenate((0, abs(hot_composite_curve["Enthalpy not Final Cumulative"]).values[:-1]), axis=None)
x=np.array(final_enthalpy_hot_streams,dtype=np.float)
y=np.array(hot_composite_curve["Temperature"],dtype=np.float)
hot_streams; z=hot_streams["Ts"].append(hot_streams["Tt"])
temp_values_of_hot_streams=np.array(z)
hot_composite_curve=pd.DataFrame()
hot_composite_curve["Temperature"]=y
hot_composite_curve["Final Enthalpy"]=x
hot_composite_curve=hot_composite_curve[hot_composite_curve['Temperature'].isin(z)]


cold_composite_curve=pd.DataFrame()
cold_composite_curve["Temperature"]=problem_table["Tleft"].values[::-1]
cold_composite_curve["Streams Involved"]=[cold_streams_involved_1[len(cold_streams_involved_1)-i-1] for i in range(len(cold_streams_involved_1))]
k=np.zeros(len(problem_table))
for i in range(len(problem_table)):
    streams=np.extract(cold_composite_curve["Streams Involved"][i]!=0, cold_composite_curve["Streams Involved"][i])
    sum_3=0
    n=len(streams)
    for n in range(len(streams)):
        sum_3=sum_3+input_table.loc[streams[n]]["FCp"]
    k[i]=sum_3
cold_composite_curve["Summation of FCp"]=k
cold_composite_curve["Difference"]=problem_table["Temperature Difference"].values[::-1]
cold_composite_curve["Enthalpy not Final"]=cold_composite_curve["Difference"]*cold_composite_curve["Summation of FCp"]
cold_composite_curve["Enthalpy not Final Cumulative"]=cold_composite_curve["Enthalpy not Final"].cumsum()
final_enthalpy_cold_streams=np.concatenate((0, abs(cold_composite_curve["Enthalpy not Final Cumulative"]).values[:-1]), axis=None)
a=np.array(final_enthalpy_cold_streams,dtype=np.float)
b=np.array(cold_composite_curve["Temperature"],dtype=np.float)
cold_streams; c=cold_streams["Ts"].append(cold_streams["Tt"])
temp_values_of_cold_streams=np.array(c)
cold_composite_curve=pd.DataFrame()
cold_composite_curve["Temperature"]=b
cold_composite_curve["Final Enthalpy"]=a+minimum_cold_utility
cold_composite_curve=cold_composite_curve[cold_composite_curve['Temperature'].isin(c)]


def Combined_Composite_Curve():
    plt.xlim(0,1.1*cold_composite_curve["Final Enthalpy"].max())
    plt.ylim(0,1.1*max(hot_composite_curve["Temperature"].max(),cold_composite_curve["Temperature"].max()))
    plt.plot(hot_composite_curve["Final Enthalpy"],hot_composite_curve["Temperature"],color='red',label='Hot Composite Curve')
    plt.plot(cold_composite_curve["Final Enthalpy"],cold_composite_curve["Temperature"],color='blue',label='Cold Composite Curve')
    plt.legend()
    plt.xlabel('Enthalpy (kW)')
    plt.ylabel('Temperature (°C)')
    plt.title('Combined Composite Curve',fontsize=20)
    plt.savefig('Combined Composite Curve.png', dpi=300, bbox_inches='tight')
    plt.show()


def Grand_Composite_Curve():
    plt.xlim(0,problem_table["Heat Flows Output"].max()+10)
    plt.ylim(0,problem_table["Tright"].max()+10)
    plt.plot(problem_table["Heat Flows Output"],problem_table["Tright"],color='blue',lw=2)
    plt.xlabel('Enthalpy(kW)')
    plt.ylabel('Temperature (°C)')
    plt.title('Grand Composite Curve',fontsize=20)
    plt.savefig('Grand Composite Curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def Pinch_in_the_grid_representation():
    plt.figure(figsize=(30,10))
    plt.ylim(-1,len(input_table)-0.5)
    plt.xlim(max(input_table["Ts"].max(),input_table["Tt"].max())+Tmin+5,min(input_table["Ts"].min(),input_table["Tt"].min())-Tmin-5) #this will also invert the x-axis limits
    plt.axvline(pinch_temp,color='black',lw=0.5,linestyle='dashed')
    for i in range(len(cold_streams)):
        for j in range(len(cold_streams),len(cold_streams)+len(hot_streams)):
            plt.arrow(hot_streams.iloc[j-len(cold_streams)]["Ts"],j,hot_streams.iloc[j-len(cold_streams)]["Tt"]-hot_streams.iloc[j-len(cold_streams)]["Ts"],0,length_includes_head=True,head_length=1.0,head_width=0.1)
            supply_hot=hot_streams.iloc[j-len(cold_streams)]["Ts"]
            plt.text(hot_streams.iloc[j-len(cold_streams)]["Ts"],j,f"{supply_hot}°",ha='right',fontsize=25)
            target_hot=hot_streams.iloc[j-len(cold_streams)]["Tt"]
            plt.text(hot_streams.iloc[j-len(cold_streams)]["Tt"],j,f"{target_hot}°",fontsize=25,ha='right')
            FCp_hot=abs(hot_streams.iloc[j-len(cold_streams)]["FCp"])
            plt.text(hot_streams.iloc[j-len(cold_streams)]["Tt"]-5,j,f"FCp={FCp_hot}kW/K",fontsize=25,fontweight='bold')
            plt.arrow(cold_streams.iloc[i]["Ts"]+Tmin,i,cold_streams.iloc[i]["Tt"]-cold_streams.iloc[i]["Ts"],0,length_includes_head=True,head_length=1.0,head_width=0.1)
            supply_cold=cold_streams.iloc[i]["Ts"]
            plt.text(cold_streams.iloc[i]["Ts"]+Tmin,i,f"{supply_cold}°",fontsize=25,ha='right')
            target_cold=cold_streams.iloc[i]["Tt"]
            plt.text(cold_streams.iloc[i]["Tt"]+Tmin,i,f"{target_cold}°",fontsize=25,ha='right')
            FCp_cold=cold_streams.iloc[i]["FCp"]
            plt.text(cold_streams.iloc[i]["Ts"]+Tmin-5,i,f"FCp={FCp_cold}kW/K",fontsize=25,fontweight='bold')
    plt.text(max(input_table["Ts"].max(),input_table["Tt"].max())+Tmin,-1,"Hot End",fontsize=20,fontweight='bold')
    plt.text(min(input_table["Ts"].min(),input_table["Tt"].min())-Tmin,-1,"Cold End",fontsize=20,ha="right",fontweight='bold')
    plt.text(pinch_temp,-1.5,"The pinch effectively divides the problem into two parts\n Heat exchange cannot take place from one end to another, otherwise there will be penalties",ha="center",fontsize=25)
    plt.xticks([])
    plt.yticks([])
    plt.title('Pinch in the grid representation',fontsize=40)
    plt.savefig('Pinch in the grid representation.png', dpi=300, bbox_inches='tight')
    plt.show()

f = interpolate.interp1d(cold_composite_curve["Final Enthalpy"], cold_composite_curve["Temperature"],bounds_error=False)
g=interpolate.interp1d(hot_composite_curve["Final Enthalpy"], hot_composite_curve["Temperature"],bounds_error=False)
extra_temp_hot=g(cold_composite_curve["Final Enthalpy"]);extra_enthalpy_hot=cold_composite_curve["Final Enthalpy"].to_numpy()
extra_temp_cold=f(hot_composite_curve["Final Enthalpy"]);extra_enthalpy_cold=hot_composite_curve["Final Enthalpy"].to_numpy()
temp={"Temperature":extra_temp_hot,"Final Enthalpy":extra_enthalpy_hot}
temp2=pd.DataFrame(temp)
hot_composite_duplicate=pd.concat([hot_composite_curve,temp2])
hot_composite_duplicate=hot_composite_duplicate.drop_duplicates()
hot_composite_duplicate=hot_composite_duplicate.sort_values('Final Enthalpy', ascending=True)
hot_composite_duplicate=hot_composite_duplicate.reset_index(drop=True)
temp={"Temperature":extra_temp_cold,"Final Enthalpy":extra_enthalpy_cold}
temp2=pd.DataFrame(temp)
cold_composite_duplicate=pd.concat([cold_composite_curve,temp2])
cold_composite_duplicate=cold_composite_duplicate.drop_duplicates()
cold_composite_duplicate=cold_composite_duplicate.sort_values('Final Enthalpy', ascending=True)
cold_composite_duplicate=cold_composite_duplicate.reset_index(drop=True)
hot_composite_duplicate = hot_composite_duplicate.rename(columns={'Temperature': 'Temperature of Hot Streams','Final Enthalpy': 'Final Enthalpy of Hot Streams'})
cold_composite_duplicate = cold_composite_duplicate.rename(columns={'Temperature': 'Temperature of Cold Streams','Final Enthalpy':'Final Enthalpy of Cold Streams'})
minimum_area_calculation=pd.concat([hot_composite_duplicate,cold_composite_duplicate],axis=1)
minimum_area_calculation=minimum_area_calculation.drop(['Final Enthalpy of Hot Streams'],axis=1)
minimum_area_calculation=minimum_area_calculation.rename(columns={'Final Enthalpy of Cold Streams':'Final Enthalpy of either streams'})
minimum_area_calculation=minimum_area_calculation.dropna()
minimum_area_calculation=minimum_area_calculation.drop_duplicates(subset='Final Enthalpy of either streams')

problem_table
Temperature_Interval_Diagram()

Combined_Composite_Curve()
Grand_Composite_Curve()

Pinch_in_the_grid_representation()
#print(input_table)


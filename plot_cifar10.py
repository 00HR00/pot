
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os



x_max = 930
x_em = 100
a1 = "Attack 2"
a2 = "Attack 3"
a5 = "Attack 4"
a6 = "Attack 1"
a0 = "Honest Training"
y_max =0.4

figsize_x = 8.5
figsize_y =5
f_s = 17
f_w = 'heavy'

current_path = os.getcwd()


lines_ele_list = []
with open('./txt/acc_cifar10.txt', "r") as file:

    content = file.read()
lines = content.split("\n")
for i in lines:
    temp = i.split("    ")
    lines_ele_list.append(temp)

lines_ele_list.pop()


cifar10 = []
cifar100 = []


for i in lines_ele_list:
    if i[2]=='CIFAR10':
        cifar10.append(i)
    elif i[2]=='CIFAR100':
        cifar100.append(i)

cifar10_resnet18 = []
cifar10_resnet34 = []
cifar10_convnet = []
cifar10_convnetd2 = []
cifar10_convnetd4 = []

cifar100_resnet18 = []
cifar100_resnet34 = []
cifar100_convnet = []
cifar100_convnetd2 = []
cifar100_convnetd4 = []

for i in cifar10:
    if i[1]=='ResNet18':
        cifar10_resnet18.append(i)
    elif i[1]=='ResNet34':
        cifar10_resnet34.append(i)
    elif i[1]=='ConvNet':
        cifar10_convnet.append(i)
    elif i[1]=='ConvNetD2':
        cifar10_convnetd2.append(i)
    elif i[1]=='ConvNetD4':
        cifar10_convnetd4.append(i)   

for i in cifar100:
    if i[1]=='ResNet18':
        cifar100_resnet18.append(i)
    elif i[1]=='ResNet34':
        cifar100_resnet34.append(i)
    elif i[1]=='ConvNet':
        cifar100_convnet.append(i)
    elif i[1]=='ConvNetD2':
        cifar100_convnetd2.append(i) 
    elif i[1]=='ConvNetD4':
        cifar100_convnetd4.append(i)


txt_dir_r = './results/'
if not os.path.exists(txt_dir_r):
    os.makedirs(txt_dir_r)








# region cifar10 resnet18
cifar10_resnet18_1 = []
cifar10_resnet18_2 = []
cifar10_resnet18_5 = []
cifar10_resnet18_6 = []
cifar10_resnet18_22 = []
cifar10_resnet18_55 = []
cifar10_resnet18_66 = []
cifar10_resnet18_honest = []
for i in cifar10_resnet18:
    if i[4]=="attacker_1":
        cifar10_resnet18_1.append(i)
    elif i[4]=="attacker_2":
        cifar10_resnet18_2.append(i)
    elif i[4]=="attacker_5":
        cifar10_resnet18_5.append(i)
    elif i[4]=="attacker_6":
        cifar10_resnet18_6.append(i)
    elif i[4]=="attacker_22":
        cifar10_resnet18_22.append(i)
    elif i[4]=="attacker_55":
        cifar10_resnet18_55.append(i)
    elif i[4]=="attacker_66":
        cifar10_resnet18_66.append(i)
    elif i[4]=="honest":
        cifar10_resnet18_honest.append(i)

cifar10_resnet18_1_x = [0]
cifar10_resnet18_1_y = [0.1]
for i in cifar10_resnet18_1:
    cifar10_resnet18_1_x.append(int(i[3])*10)
    cifar10_resnet18_1_y.append(float(i[0]))


cifar10_resnet18_2_x = [0]
cifar10_resnet18_2_y = [0.1]
for i in cifar10_resnet18_2:
    cifar10_resnet18_2_x.append(int(i[3])*10)
    cifar10_resnet18_2_y.append(float(i[0]))

cifar10_resnet18_5_x = [0]
cifar10_resnet18_5_y = [0.1]
for i in cifar10_resnet18_5:
    cifar10_resnet18_5_x.append(int(i[3])*10)
    cifar10_resnet18_5_y.append(float(i[0]))

cifar10_resnet18_6_x = [0]
cifar10_resnet18_6_y = [0.1]
for i in cifar10_resnet18_6:
    cifar10_resnet18_6_x.append(int(i[3])*10)
    cifar10_resnet18_6_y.append(float(i[0]))

cifar10_resnet18_22_x = [0]
cifar10_resnet18_22_y = [0.1]
for i in cifar10_resnet18_22:
    cifar10_resnet18_22_x.append(int(i[3])*10)
    cifar10_resnet18_22_y.append(float(i[0]))

cifar10_resnet18_55_x = [0]
cifar10_resnet18_55_y = [0.1]
for i in cifar10_resnet18_55:
    cifar10_resnet18_55_x.append(int(i[3])*10)
    cifar10_resnet18_55_y.append(float(i[0]))

cifar10_resnet18_66_x = [0]
cifar10_resnet18_66_y = [0.1]
for i in cifar10_resnet18_66:
    cifar10_resnet18_66_x.append(int(i[3])*10)
    cifar10_resnet18_66_y.append(float(i[0]))

cifar10_resnet18_honest_x = [0]
cifar10_resnet18_honest_y = [0.1]
for i in cifar10_resnet18_honest:
    cifar10_resnet18_honest_x.append(int(i[3])*10)
    cifar10_resnet18_honest_y.append(float(i[0]))



font1 = {'family' : 'Times New Roman',
'weight' : f_w,
'size'   : f_s,
}


fig = plt.figure(num=1,figsize=(figsize_x,figsize_y))
plt.xticks(np.arange(0, x_max, x_em),fontproperties = 'Times New Roman', size = 12)
plt.xlim(0, x_max)
plt.title('ResNet18')

legend_elements = [
                   Line2D([0], [0], marker='D', color='#1FCA7D', label=a0,
                          markerfacecolor='#1FCA7D', markersize=7), 
                   Line2D([0], [0], marker='^', color='#FBC228', label=a6,
                          markerfacecolor='#FBC228', markersize=7),
                   Line2D([0], [0], marker='o', color='#0000CD', label=a1,
                          markerfacecolor='#0000CD', markersize=7),
                   Line2D([0], [0], marker='s', color='#F54848', label=a2,
                          markerfacecolor='#F54848', markersize=7),
                   Line2D([0], [0], marker='x', color='#01CAFF', label=a5,
                          markerfacecolor='#01CAFF', markersize=7),
                   Line2D([0], [0], color='black', lw=2, linestyle=':', label='Random Guess'),
                   Line2D([0], [0], color='black', lw=2, linestyle='--', label='Shorter Trajectory'),
                   Line2D([0], [0], color='black', lw=2, linestyle='-', label='Longer Trajectory'),]

plt.axhline(y=0.1, xmin=0, xmax=93, linestyle=':',color = "black")
plt.plot(cifar10_resnet18_1_x, cifar10_resnet18_1_y, marker = 'o', color='#0000CD', label=a1,linestyle='-',linewidth=2)
plt.plot(cifar10_resnet18_2_x, cifar10_resnet18_2_y, marker = 's', color='#F54848', label=a2,linestyle='--',linewidth=2)
plt.plot(cifar10_resnet18_22_x, cifar10_resnet18_22_y, marker = 's', color='#F54848', label=a2,linestyle='-',linewidth=2)
plt.plot(cifar10_resnet18_5_x, cifar10_resnet18_5_y, marker = 'x', color='#01CAFF', label=a5,linestyle='-',linewidth=2)
plt.plot(cifar10_resnet18_55_x, cifar10_resnet18_55_y, marker = 'x', color='#01CAFF', label=a5,linestyle='--',linewidth=2)
plt.plot(cifar10_resnet18_6_x, cifar10_resnet18_6_y, marker = '^', color='#FBC228', label=a6,linestyle='-',linewidth=2)
plt.plot(cifar10_resnet18_66_x, cifar10_resnet18_66_y, marker = '^', color='#FBC228', label=a6,linestyle='--',linewidth=2)
plt.plot(cifar10_resnet18_honest_x, cifar10_resnet18_honest_y, marker = 'D', color='#1FCA7D', label=a0,linestyle='-',linewidth=2)


plt.legend(handles=legend_elements, prop=font1,loc = 'upper center',bbox_to_anchor=(0.78,1))

plt.xlabel('Synthetic Data Size (SDS)',size='16')
plt.ylabel('Average Accuracy',size='16')
plt.show()

fig.savefig("./results/cifar10_ResNet18.pdf",dpi=1200)
#endregion











#region cifar10 resnet34
cifar10_resnet34_1 = []
cifar10_resnet34_2 = []
cifar10_resnet34_5 = []
cifar10_resnet34_6 = []
cifar10_resnet34_22 = []
cifar10_resnet34_55 = []
cifar10_resnet34_66 = []
cifar10_resnet34_honest = []
for i in cifar10_resnet34:
    if i[4]=="attacker_1":
        cifar10_resnet34_1.append(i)
    elif i[4]=="attacker_2":
        cifar10_resnet34_2.append(i)
    elif i[4]=="attacker_5":
        cifar10_resnet34_5.append(i)
    elif i[4]=="attacker_6":
        cifar10_resnet34_6.append(i)
    elif i[4]=="attacker_22":
        cifar10_resnet34_22.append(i)
    elif i[4]=="attacker_55":
        cifar10_resnet34_55.append(i)
    elif i[4]=="attacker_66":
        cifar10_resnet34_66.append(i)
    elif i[4]=="honest":
        cifar10_resnet34_honest.append(i)


cifar10_resnet34_1_x = [0]
cifar10_resnet34_1_y = [0.1]
for i in cifar10_resnet34_1:
    cifar10_resnet34_1_x.append(int(i[3])*10)
    cifar10_resnet34_1_y.append(float(i[0]))
                                
cifar10_resnet34_2_x = [0]
cifar10_resnet34_2_y = [0.1]
for i in cifar10_resnet34_2:
    cifar10_resnet34_2_x.append(int(i[3])*10)
    cifar10_resnet34_2_y.append(float(i[0]))

cifar10_resnet34_5_x = [0]
cifar10_resnet34_5_y = [0.1]
for i in cifar10_resnet34_5:
    cifar10_resnet34_5_x.append(int(i[3])*10)
    cifar10_resnet34_5_y.append(float(i[0]))

cifar10_resnet34_6_x = [0]
cifar10_resnet34_6_y = [0.1]
for i in cifar10_resnet34_6:
    cifar10_resnet34_6_x.append(int(i[3])*10)
    cifar10_resnet34_6_y.append(float(i[0]))

cifar10_resnet34_22_x = [0]
cifar10_resnet34_22_y = [0.1]
for i in cifar10_resnet34_22:
    cifar10_resnet34_22_x.append(int(i[3])*10)
    cifar10_resnet34_22_y.append(float(i[0]))

cifar10_resnet34_55_x = [0]
cifar10_resnet34_55_y = [0.1]
for i in cifar10_resnet34_55:
    cifar10_resnet34_55_x.append(int(i[3])*10)
    cifar10_resnet34_55_y.append(float(i[0]))

cifar10_resnet34_66_x = [0]
cifar10_resnet34_66_y = [0.1]
for i in cifar10_resnet34_66:
    cifar10_resnet34_66_x.append(int(i[3])*10)
    cifar10_resnet34_66_y.append(float(i[0]))

cifar10_resnet34_honest_x = [0]
cifar10_resnet34_honest_y = [0.1]
for i in cifar10_resnet34_honest:
    cifar10_resnet34_honest_x.append(int(i[3])*10)
    cifar10_resnet34_honest_y.append(float(i[0]))


font1 = {'family' : 'Times New Roman',
'weight' : f_w,
'size'   : f_s,
}

fig = plt.figure(num=1,figsize=(figsize_x,figsize_y))
plt.xticks(np.arange(0, x_max, x_em),fontproperties = 'Times New Roman', size = 12)
plt.xlim(0, x_max)
plt.title('ResNet34')

legend_elements = [
                   Line2D([0], [0], marker='D', color='#1FCA7D', label=a0,
                          markerfacecolor='#1FCA7D', markersize=7), 
                   Line2D([0], [0], marker='^', color='#FBC228', label=a6,
                          markerfacecolor='#FBC228', markersize=7),
                   Line2D([0], [0], marker='o', color='#0000CD', label=a1,
                          markerfacecolor='#0000CD', markersize=7),
                   Line2D([0], [0], marker='s', color='#F54848', label=a2,
                          markerfacecolor='#F54848', markersize=7),
                   Line2D([0], [0], marker='x', color='#01CAFF', label=a5,
                          markerfacecolor='#01CAFF', markersize=7),
                   Line2D([0], [0], color='black', lw=2, linestyle=':', label='Random Guess'),
                   Line2D([0], [0], color='black', lw=2, linestyle='--', label='Shorter Trajectory'),
                   Line2D([0], [0], color='black', lw=2, linestyle='-', label='Longer Trajectory'),]

plt.axhline(y=0.1, xmin=0, xmax=93, linestyle=':',color = "black")
plt.plot(cifar10_resnet34_1_x, cifar10_resnet34_1_y, marker = 'o', color='#0000CD', label=a1,linestyle='-',linewidth=2)
plt.plot(cifar10_resnet34_2_x, cifar10_resnet34_2_y, marker = 's', color='#F54848', label=a2,linestyle='--',linewidth=2)
plt.plot(cifar10_resnet34_22_x, cifar10_resnet34_22_y, marker = 's', color='#F54848', label=a2,linestyle='-',linewidth=2)
plt.plot(cifar10_resnet34_5_x, cifar10_resnet34_5_y, marker = 'x', color='#01CAFF', label=a5,linestyle='-',linewidth=2)
plt.plot(cifar10_resnet34_55_x, cifar10_resnet34_55_y, marker = 'x', color='#01CAFF', label=a5,linestyle='--',linewidth=2)
plt.plot(cifar10_resnet34_6_x, cifar10_resnet34_6_y, marker = '^', color='#FBC228', label=a6,linestyle='-',linewidth=2)
plt.plot(cifar10_resnet34_66_x, cifar10_resnet34_66_y, marker = '^', color='#FBC228', label=a6,linestyle='--',linewidth=2)
plt.plot(cifar10_resnet34_honest_x, cifar10_resnet34_honest_y, marker = 'D', color='#1FCA7D', label=a0,linestyle='-',linewidth=2)

plt.legend(handles=legend_elements, prop=font1,loc = 'upper center',bbox_to_anchor=(0.78,1))

plt.xlabel('Synthetic Data Size (SDS)',size='16')
plt.ylabel('Average Accuracy',size='16')
plt.show()
fig.savefig("./results/cifar10_ResNet34.pdf",dpi=1200)
#endregion






# region cifar10 convnetd2
cifar10_convnetd2_1 = []
cifar10_convnetd2_2 = []
cifar10_convnetd2_5 = []
cifar10_convnetd2_6 = []
cifar10_convnetd2_22 = []
cifar10_convnetd2_55 = []
cifar10_convnetd2_66 = []
cifar10_convnetd2_honest = []
for i in cifar10_convnetd2:
    if i[4]=="attacker_1":
        cifar10_convnetd2_1.append(i)
    elif i[4]=="attacker_2":
        cifar10_convnetd2_2.append(i)
    elif i[4]=="attacker_5":
        cifar10_convnetd2_5.append(i)
    elif i[4]=="attacker_6":
        cifar10_convnetd2_6.append(i)
    elif i[4]=="attacker_22":
        cifar10_convnetd2_22.append(i)
    elif i[4]=="attacker_55":
        cifar10_convnetd2_55.append(i)
    elif i[4]=="attacker_66":
        cifar10_convnetd2_66.append(i)
    elif i[4]=="honest":
        cifar10_convnetd2_honest.append(i)


cifar10_convnetd2_1_x = [0]
cifar10_convnetd2_1_y = [0.1]
for i in cifar10_convnetd2_1:
    cifar10_convnetd2_1_x.append(int(i[3])*10)
    cifar10_convnetd2_1_y.append(float(i[0]))
                                
cifar10_convnetd2_2_x = [0]
cifar10_convnetd2_2_y = [0.1]
for i in cifar10_convnetd2_2:
    cifar10_convnetd2_2_x.append(int(i[3])*10)
    cifar10_convnetd2_2_y.append(float(i[0]))

cifar10_convnetd2_5_x = [0]
cifar10_convnetd2_5_y = [0.1]
for i in cifar10_convnetd2_5:
    cifar10_convnetd2_5_x.append(int(i[3])*10)
    cifar10_convnetd2_5_y.append(float(i[0]))

cifar10_convnetd2_6_x = [0]
cifar10_convnetd2_6_y = [0.1]
for i in cifar10_convnetd2_6:
    cifar10_convnetd2_6_x.append(int(i[3])*10)
    cifar10_convnetd2_6_y.append(float(i[0]))

cifar10_convnetd2_22_x = [0]
cifar10_convnetd2_22_y = [0.1]
for i in cifar10_convnetd2_22:
    cifar10_convnetd2_22_x.append(int(i[3])*10)
    cifar10_convnetd2_22_y.append(float(i[0]))

cifar10_convnetd2_55_x = [0]
cifar10_convnetd2_55_y = [0.1]
for i in cifar10_convnetd2_55:
    cifar10_convnetd2_55_x.append(int(i[3])*10)
    cifar10_convnetd2_55_y.append(float(i[0]))

cifar10_convnetd2_66_x = [0]
cifar10_convnetd2_66_y = [0.1]
for i in cifar10_convnetd2_66:
    cifar10_convnetd2_66_x.append(int(i[3])*10)
    cifar10_convnetd2_66_y.append(float(i[0]))

cifar10_convnetd2_honest_x = [0]
cifar10_convnetd2_honest_y = [0.1]
for i in cifar10_convnetd2_honest:
    cifar10_convnetd2_honest_x.append(int(i[3])*10)
    cifar10_convnetd2_honest_y.append(float(i[0]))


font1 = {'family' : 'Times New Roman',
'weight' : f_w,
'size'   : f_s,
}

fig = plt.figure(num=1,figsize=(figsize_x,figsize_y))
plt.xticks(np.arange(0, x_max, x_em),fontproperties = 'Times New Roman', size = 12)
plt.xlim(0, x_max)
plt.title('ConvNetD2')

legend_elements = [
                   Line2D([0], [0], marker='D', color='#1FCA7D', label=a0,
                          markerfacecolor='#1FCA7D', markersize=7), 
                   Line2D([0], [0], marker='^', color='#FBC228', label=a6,
                          markerfacecolor='#FBC228', markersize=7),
                   Line2D([0], [0], marker='o', color='#0000CD', label=a1,
                          markerfacecolor='#0000CD', markersize=7),
                   Line2D([0], [0], marker='s', color='#F54848', label=a2,
                          markerfacecolor='#F54848', markersize=7),
                   Line2D([0], [0], marker='x', color='#01CAFF', label=a5,
                          markerfacecolor='#01CAFF', markersize=7),
                   Line2D([0], [0], color='black', lw=2, linestyle=':', label='Random Guess'),
                   Line2D([0], [0], color='black', lw=2, linestyle='--', label='Shorter Trajectory'),
                   Line2D([0], [0], color='black', lw=2, linestyle='-', label='Longer Trajectory'),]

plt.axhline(y=0.1, xmin=0, xmax=93, linestyle=':',color = "black")
plt.plot(cifar10_convnetd2_1_x, cifar10_convnetd2_1_y, marker = 'o', color='#0000CD', label=a1,linestyle='-',linewidth=2)
plt.plot(cifar10_convnetd2_2_x, cifar10_convnetd2_2_y, marker = 's', color='#F54848', label=a2,linestyle='--',linewidth=2)
plt.plot(cifar10_convnetd2_22_x, cifar10_convnetd2_22_y, marker = 's', color='#F54848', label=a2,linestyle='-',linewidth=2)
plt.plot(cifar10_convnetd2_5_x, cifar10_convnetd2_5_y, marker = 'x', color='#01CAFF', label=a5,linestyle='-',linewidth=2)
plt.plot(cifar10_convnetd2_55_x, cifar10_convnetd2_55_y, marker = 'x', color='#01CAFF', label=a5,linestyle='--',linewidth=2)
plt.plot(cifar10_convnetd2_6_x, cifar10_convnetd2_6_y, marker = '^', color='#FBC228', label=a6,linestyle='-',linewidth=2)
plt.plot(cifar10_convnetd2_66_x, cifar10_convnetd2_66_y, marker = '^', color='#FBC228', label=a6,linestyle='--',linewidth=2)
plt.plot(cifar10_convnetd2_honest_x, cifar10_convnetd2_honest_y, marker = 'D', color='#1FCA7D', label=a0,linestyle='-',linewidth=2)

plt.legend(handles=legend_elements, prop=font1,loc = 'upper center',bbox_to_anchor=(0.78,1))

plt.xlabel('Synthetic Data Size (SDS)',size='16')
plt.ylabel('Average Accuracy',size='16')
plt.show()
fig.savefig("./results/cifar10_CNN2D.pdf",dpi=1200)
#endregion
       












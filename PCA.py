import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import random
import pandas as pd
from sympy import *

plt.figure(figsize=(10,10))

omega1=[[1,0],[3,2],[2,-1],[-1,-2],[0,1],[-1,0],[1,2],[0,-1],[-3,-2],[-2,1]]
omega2=[[-1,0],[1,2],[0,-1],[-3,-2],[-2,1]]
#omega3=omega1.append(omega2)
#print(omega3)
omega1v=np.array(omega1)
omega2v=np.array(omega2)
df = pd.DataFrame(omega1v)
cov1=np.cov(omega1v.T, bias=True)
print(cov1)
print(f"eigenvectors are \n {(np.linalg.eig(cov1))[1]}")
a = (np.linalg.eig(cov1))[1][0][0]
b = (np.linalg.eig(cov1))[1][1][0]

#print(a, b)
print(f"w1, w2 of the points are {round((np.linalg.eig(cov1))[0][0],3)} and {(np.linalg.eig(cov1))[0][1]}")

ax = plt.gca()    # 得到图像的Axes对象
ax.spines['right'].set_color('none')   # 将图像右边的轴设为透明
ax.spines['top'].set_color('none')     # 将图像上面的轴设为透明
ax.xaxis.set_ticks_position('bottom')    # 将x轴刻度设在下面的坐标轴上
ax.yaxis.set_ticks_position('left')         # 将y轴刻度设在左边的坐标轴上
ax.spines['bottom'].set_position(('data', 0))   # 将两个坐标轴的位置设在数据点原点
ax.spines['left'].set_position(('data', 0))

x1 = np.linspace(-4,4)
y1 = (-a/b) *x1

x2 = np.linspace(-4,4)
y2 = (b/a) *x2

plt.plot(x1,y1,label='lamda1')
plt.plot(x2,y2)
plt.axis('image')

plt.scatter(omega1v[:,0],omega1v[:,1])
plt.scatter(omega2v[:,0],omega2v[:,1],marker='>')
# plt.plot(x1,y1)
plt.legend(['lamda1','lamda2','class1&2'])




plt.show()

def calcub(x,y): #降维，即求原来的二维点，在新坐标系中的，拥有较大方差的坐标轴上的，一维投影距离
    newdimension = x*a + y*b
    print(round(newdimension,3))

print("The new dimensions for point 1 to point 10 are")    
i=0 
while(i<len(omega1)):
    x3=omega1[i][0]
    y3=omega1[i][1]
    calcub(x3,y3)
    i=i+1

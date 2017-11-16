# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:17:13 2015

@author: Eddy_zheng
"""
"""
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()
"""

import matplotlib.pyplot as plt #绘图用的模块  
from mpl_toolkits.mplot3d import Axes3D #绘制3D坐标的函数  
import numpy as np  
  
def fun(x,y):  
    return np.power(x,2)+np.power(y,2)  
  
fig1=plt.figure()#创建一个绘图对象  
ax=Axes3D(fig1)#用这个绘图对象创建一个Axes对象(有3D坐标)  
X=np.arange(-2,2,0.1)  
Y=np.arange(-2,2,0.1)#创建了从-2到2，步长为0.1的arange对象  
#至此X,Y分别表示了取样点的横纵坐标的可能取值  
#用这两个arange对象中的可能取值一一映射去扩充为所有可能的取样点  
X,Y=np.meshgrid(X,Y)  
Z=fun(X,Y)#用取样点横纵坐标去求取样点Z坐标  
plt.title("This is main title")#总标题  
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)#用取样点(x,y,z)去构建曲面  
ax.set_xlabel('x label', color='r')  
ax.set_ylabel('y label', color='g')  
ax.set_zlabel('z label', color='b')#给三个坐标轴注明  
plt.show()#显示模块中的所有绘图对象  

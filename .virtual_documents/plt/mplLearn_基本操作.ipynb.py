# get_ipython().run_line_magic("matplotlib", " qt5")
# get_ipython().run_line_magic("matplotlib", " inline")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.pyplot.ion() #交互式开启



#figsize(12.5, 4) # 设置 figsize
# plt.rcParams['savefig.dpi'] = 300 #图片像素
# plt.rcParams['figure.dpi'] = 300 #分辨率
# 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
# 指定dpi=200，图片尺寸为 1200*800
# 指定dpi=300，图片尺寸为 1800*1200
# 设置figsize可以在不改变分辨率情况下改变比例



x=np.linspace(-3,3,50)
y1=2*x+1
y2=x**2
y3=0.1*x


plt.figure()
plt.plot(x,y1)
plt.show()


plt.figure()
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='-.')
plt.show()


#设置坐标轴

plt.figure()
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='-.')

plt.xlim((-1,2)) #设置坐标轴的显示范围
plt.ylim((-2,3))

plt.xlabel('I am xget_ipython().getoutput("') #改变坐标轴标签")
plt.ylabel('I am yget_ipython().getoutput("')")

new_ticks=np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)  #更换坐标轴数值
plt.yticks([-2,-1,0,1,2,3],
          ['bad','little bad','normal','soso','little good','good']) #可以用latex公式语法改变字体、添加符号

ax=plt.gca() #gca=get current axis，指上下左右四个边框
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom') #选择下边的轴为x坐标轴
ax.yaxis.set_ticks_position('left') #选择左边的轴为y坐标轴
ax.spines['bottom'].set_position(('data',0)) #设置位置 ‘data:数值位置’'axes:百分比位置'
ax.spines['left'].set_position(('data',0))

plt.show()


#legend图例

plt.figure()
# plt.plot(x,y2,label='up')
# plt.plot(x,y1,color='red',linewidth=1.0,linestyle='-.',label='down')
l1,=plt.plot(x,y2,label='up')  #如果在图例里要对此进行选择更改，命名的时候要加逗号','
l2,=plt.plot(x,y1,color='red',linewidth=1.0,linestyle='-.',label='down')

#显示图例。handles:选择显示图例的曲线。labels:对应的图例内容。loc:图例位置,'best'为自动。
plt.legend(handles=[l1,l2],labels=['a','b'],loc='best')


plt.show()


#annotation注释
plt.figure(num=1,figsize=(8,5))
plt.plot(x,y1)
# plt.scatter(x,y) #散点图

ax=plt.gca() 
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom') 
ax.yaxis.set_ticks_position('left') 
ax.spines['bottom'].set_position(('data',0)) 
ax.spines['left'].set_position(('data',0))

#画点
x0=1
y0=2*x0+1
plt.scatter(x0,y0,s=50,color='b')  #s:size
plt.plot([x0,x0],[y0,0],'k--',lw=2.5) #坐标是竖着看的，第一个大括号是两个点的x坐标

#数据+箭头
#xy:目标点  xytext:偏移量
plt.annotate(s='2x+1={}'.format(y0),xy=(x0,y0),xycoords='data',xytext=(+30,-30),textcoords='offset points',fontsize=16,color='r',
            arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))

#文本
plt.text(-3.7,3,r'$This\ is\ the\ some\ text.\ \mu\ \sigma_i\ \alpha_t$',
        fontdict={'size':16,'color':'r'})

plt.show()


#tick 能见度

plt.figure()
plt.plot(x,y3,linewidth=10,zorder=1) #zuoder为叠加高度
plt.ylim(-2,2)
ax=plt.gca() 
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom') 
ax.yaxis.set_ticks_position('left') 
ax.spines['bottom'].set_position(('data',0)) 
ax.spines['left'].set_position(('data',0))

for label in ax.get_xticklabels()+ax.get_yticklabels():
    label.set_zorder(2)
    label.set_fontsize(12) #字体大小
    label.set_bbox(dict(facecolor='red',edgecolor='None',alpha=0.3)) #为标签添加边框  alpha:透明度

plt.show()

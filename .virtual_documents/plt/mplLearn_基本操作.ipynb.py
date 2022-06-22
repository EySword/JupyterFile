# get_ipython().run_line_magic("matplotlib", " qt5")
# get_ipython().run_line_magic("matplotlib", " inline")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.pyplot.ion() #����ʽ����



#figsize(12.5, 4) # ���� figsize
# plt.rcParams['savefig.dpi'] = 300 #ͼƬ����
# plt.rcParams['figure.dpi'] = 300 #�ֱ���
# Ĭ�ϵ����أ�[6.0,4.0]���ֱ���Ϊ100��ͼƬ�ߴ�Ϊ 600&400
# ָ��dpi=200��ͼƬ�ߴ�Ϊ 1200*800
# ָ��dpi=300��ͼƬ�ߴ�Ϊ 1800*1200
# ����figsize�����ڲ��ı�ֱ�������¸ı����



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


#����������

plt.figure()
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='-.')

plt.xlim((-1,2)) #�������������ʾ��Χ
plt.ylim((-2,3))

plt.xlabel('I am xget_ipython().getoutput("') #�ı��������ǩ")
plt.ylabel('I am yget_ipython().getoutput("')")

new_ticks=np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)  #������������ֵ
plt.yticks([-2,-1,0,1,2,3],
          ['bad','little bad','normal','soso','little good','good']) #������latex��ʽ�﷨�ı����塢��ӷ���

ax=plt.gca() #gca=get current axis��ָ���������ĸ��߿�
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom') #ѡ���±ߵ���Ϊx������
ax.yaxis.set_ticks_position('left') #ѡ����ߵ���Ϊy������
ax.spines['bottom'].set_position(('data',0)) #����λ�� ��data:��ֵλ�á�'axes:�ٷֱ�λ��'
ax.spines['left'].set_position(('data',0))

plt.show()


#legendͼ��

plt.figure()
# plt.plot(x,y2,label='up')
# plt.plot(x,y1,color='red',linewidth=1.0,linestyle='-.',label='down')
l1,=plt.plot(x,y2,label='up')  #�����ͼ����Ҫ�Դ˽���ѡ����ģ�������ʱ��Ҫ�Ӷ���','
l2,=plt.plot(x,y1,color='red',linewidth=1.0,linestyle='-.',label='down')

#��ʾͼ����handles:ѡ����ʾͼ�������ߡ�labels:��Ӧ��ͼ�����ݡ�loc:ͼ��λ��,'best'Ϊ�Զ���
plt.legend(handles=[l1,l2],labels=['a','b'],loc='best')


plt.show()


#annotationע��
plt.figure(num=1,figsize=(8,5))
plt.plot(x,y1)
# plt.scatter(x,y) #ɢ��ͼ

ax=plt.gca() 
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom') 
ax.yaxis.set_ticks_position('left') 
ax.spines['bottom'].set_position(('data',0)) 
ax.spines['left'].set_position(('data',0))

#����
x0=1
y0=2*x0+1
plt.scatter(x0,y0,s=50,color='b')  #s:size
plt.plot([x0,x0],[y0,0],'k--',lw=2.5) #���������ſ��ģ���һ�����������������x����

#����+��ͷ
#xy:Ŀ���  xytext:ƫ����
plt.annotate(s='2x+1={}'.format(y0),xy=(x0,y0),xycoords='data',xytext=(+30,-30),textcoords='offset points',fontsize=16,color='r',
            arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))

#�ı�
plt.text(-3.7,3,r'$This\ is\ the\ some\ text.\ \mu\ \sigma_i\ \alpha_t$',
        fontdict={'size':16,'color':'r'})

plt.show()


#tick �ܼ���

plt.figure()
plt.plot(x,y3,linewidth=10,zorder=1) #zuoderΪ���Ӹ߶�
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
    label.set_fontsize(12) #�����С
    label.set_bbox(dict(facecolor='red',edgecolor='None',alpha=0.3)) #Ϊ��ǩ��ӱ߿�  alpha:͸����

plt.show()

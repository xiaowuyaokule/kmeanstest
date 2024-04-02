'''
这是一个K-均值聚类算法的GUI界面下的演示程序，采用199个银行客户数据集（年龄，存款），聚类结束后给出后处理的银行客户画像

'''
import cv2
import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import tkinter as tk
import tkinter.filedialog as file
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # 导入在tkinter 内嵌入的画布子窗口
from sklearn.cluster import KMeans

from KmeansTest5 import ax0


MainWin = tk.Tk()  # 建立主窗口
MainWin.title('机器学习K-均值算法演示')  # 窗口标题
MainWin.geometry('1200x700')  # 窗口宽X高尺寸
MainWin.resizable(width=True, height=True)  # 设定窗口尺寸可变

# Kmeans算法演示程序-银行客户画像
#data全局变量数据集
data = []
#聚类类别数
K = 3
"""
    K-means聚类算法
    参数：
    X: numpy数组，形状为 (样本数, 特征数)，表示待聚类的数据集
    k: int，聚类的簇数
    max_iters: int，最大迭代次数，默认为100
    返回值：
    centroids: numpy数组，形状为 (k, 特征数)，表示每个簇的中心点
    labels: numpy数组，形状为 (样本数,)，表示每个样本所属的簇
"""

def Kmeans(X, k, max_iters=100):
    # 随机初始化中心点
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for N in range(max_iters):
        # 计算每个样本到中心点的距离
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        #print(distances)
        # 分配样本到最近的中心点所在的簇
        labels = np.argmin(distances, axis=1)
        #print("labels", labels)
        # 更新中心点为每个簇的均值
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        # 如果中心点没有变化，提前结束迭代
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        #print(centroids)
        #print("\nN=",N)
    return centroids, labels
#三种不同的聚类数
Num = ((16, 6, 2),(30, 8, 2),(60, 10, 2))
#选取原图像的比例
Percent = 0.3
#调用SKlearn 的聚类函数进行图像聚类
def CallKmeans():
    global data, K, Percent
    ResultStr.set("开始聚类")
    MainWin.update()
    X = data.reshape(-1, 3)
    # KMeans聚类
    n_colors=Num[K % 3]
    imgs = []
    N = X.shape[0]
    for n_cluster in n_colors:
        idx = np.random.randint(0, N, size=int(Percent*N))
        x = X[idx]
        km = KMeans(n_clusters=n_cluster) #构建模型
        km.fit(x)# 拟合训练数据
        result = km.predict(X) #预测结果
        img = km.cluster_centers_[result]
        imgs.append(img.reshape(data.shape))

    # 可视化展示
    ax1.imshow(data.astype('uint8'))
    ax2.imshow(imgs[0].astype('uint8'))
    ax3.imshow(imgs[1].astype('uint8'))
    ax4.imshow(imgs[2].astype('uint8'))
    ax1.set_title('Raw image')
    ax2.set_title('Cluster num:'+str(n_colors[0]))
    ax3.set_title('Cluster num:'+str(n_colors[1]))
    ax4.set_title('Cluster num:'+str(n_colors[2]))
    draw_set.draw()
    ResultStr.set("图像聚类完成")
    return

def callback():
    print("被调用了")
    return

# 在logo区显示一幅图像的函数
ww = 500
wh = 400

def picshow(filename):
    I1 = cv2.imread(filename)
    I2 = cv2.resize(I1, (ww, wh))
    cv2.imwrite('ImageFile/temp.png', I2)
    I3 = tk.PhotoImage(file='ImageFile/temp.png')
    L1 = tk.Label(InputFrame, image=I3)
    L1.grid(row=1, column=0, columnspan=2, padx=40)
    MainWin.mainloop()
    return
#设置K均值的K值 范围：2-9
def SetKvalue():
    global K
    k = TargetV.get()
    if 1 < k < 10:
        K = k
        EpochStr.set(str(K))

#装入鸢尾花数据集
def LoadIris():
    global data
    data1 = pd.read_csv('iris2.csv')  # 训练数据集
    data = np.array(data1.iloc[0:, 1:3])
    CountStr.set(str(len(data)))
    EpochStr.set(str(K))
    but5.config(state=tk.ACTIVE)

#装入物流坐标
def LoadCoordinate():
    global data
    data = np.loadtxt('testSet.txt')  # 训练数据集
    print(data)
    #ax.set_autoscale_on(False)  # 设置自动量程
    cmin=data.min(0)
    cmax=data.max(0)
    img = cv2.imread('city1.jpg')
    ax0.clear()
    ax1.clear()
    ax0.imshow(img)
    ax0.set_axis_off()
    draw_set.draw()
    ax1.set_xlim([cmin[0], cmax[0]])
    ax1.set_ylim([cmin[1], cmax[1]])
    X = data
    ax1.set_axis_on()
    ax1.scatter(X[:, 0], X[:, 1], c="b")
    draw_set.draw()
    # data = np.array(data1.iloc[0:, 1:3])
    CountStr.set(str(len(data)))
    EpochStr.set(str(K))
    but5.config(state=tk.ACTIVE)
def LoadCustomerInfo():
    global data
    data1 = pd.read_csv('CustomerInfo.csv')  # 训练数据集
    data=np.array(data1.iloc[0:,3:5])
    print(data)
    #ax.set_autoscale_on(False)  # 设置自动量程
    cmin=data.min(0)
    cmax=data.max(0)
    ax1.clear()
    ax1.set_xlim([cmin[0], cmax[0]])
    ax1.set_ylim([cmin[1], cmax[1]])
    X = data
    ax1.set_axis_on()
    ax1.scatter(X[:, 0], X[:, 1], c="b")
    draw_set.draw()
    # data = np.array(data1.iloc[0:, 1:3])
    CountStr.set(str(len(data)))
    EpochStr.set(str(K))
    but5.config(state=tk.ACTIVE)
    mmenu.entryconfig("K-均值聚类算法", state=tk.ACTIVE)
ImageFile=""
def LoadImage():
    global data
    file1 = file.askopenfilename(title='打开文件名字', initialdir="\image",
                                 filetypes=[('jpg文件', '*.jpg'),('png文件', '*.png')])
    if file1 == "":
        return
    ImageFile=file1
    im = cv2.imread(ImageFile)
    h=im.shape[0]
    w=im.shape[1]
    im1=cv2.resize(im,[int(300*h/w),300])
    image = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)  # 转换成RGB格式
    data=image
    but5.config(state=tk.ACTIVE)
    picshow(ImageFile)
    return
def LoadSample():
    global data, K
    data = np.array([
        [7, 11],
        [9, 4],
        [13, 6],
        [6, 5],
        [7, 6],
        [5, 6],
        [3, 9],
        [13, 5],
        [4, 11],
        [4, 4],
        [5, 1],
        [8.3, 4.2],
        [7.5, 1.0],
        [2, 2],
        [3.7, 2.6]
    ])
    CountStr.set(str(len(data)))
    EpochStr.set(str(K))
    but5.config(state=tk.ACTIVE)


# =========菜单区=============
menubar = tk.Menu(MainWin)
# file menu
fmenu = tk.Menu(menubar)  # 主菜单上加入下级子菜单
fmenu.add_command(label='新建', state=tk.DISABLED, command=callback)
fmenu.add_command(label='打开', state=tk.DISABLED, command=callback)
fmenu.add_command(label='保存', state=tk.DISABLED, command=callback)
fmenu.add_command(label='另存为', state=tk.DISABLED, command=callback)
# Image processing menu
#imenu = tk.Menu(menubar)
#imenu.add_command(label='图像处理', state=tk.DISABLED, command=callback)
#imenu.add_command(label='图像跟踪', state=tk.DISABLED, command=callback)
#imenu.add_command(label='人脸检测', state=tk.DISABLED, command=callback)
#imenu.add_command(label='图像爬取', state=tk.DISABLED, command=callback)
# machine learning
mmenu = tk.Menu(menubar)
mmenu.add_command(label='KNN-分类算法', state=tk.DISABLED, command=callback)
mmenu.add_command(label='K-均值聚类算法', state=tk.DISABLED, command=CallKmeans)
mmenu.add_command(label='支持向量机', state=tk.DISABLED, command=callback)
mmenu.add_command(label='BP神经网络', state=tk.DISABLED, command=callback)
mmenu.add_command(label='CNN卷积神经网', state=tk.DISABLED, command=callback)
# =============
menubar.add_cascade(label="文件操作", menu=fmenu)
#menubar.add_cascade(label="图像处理", menu=imenu)
menubar.add_cascade(label="机器学习", menu=mmenu)
MainWin.config(menu=menubar)
#mmenu.entryconfig("K-均值聚类算法", state=tk.DISABLED)

# tkinter窗口的布局
# 设置“田”字型4个布局框架Frame 区
# InputFrame：logo图片显示，和识别结果显示
# OutputFrame：曲线输出，文本输出
# butFrame：按钮区用于放置各种功能按钮，用组件Button做执行选择，Radiobutton做排它选择
# DataFrame：用于显示各种数据，一般用组件 Entry输出数据, 组件Label命名（贴标签）
InputFrame = tk.Frame(MainWin, height=300, width=600, relief=tk.RAISED)
OutputFrame = tk.Frame(MainWin, height=300, width=600, relief=tk.RAISED)
butFrame = tk.Frame(MainWin, height=300, width=600)
DataFrame = tk.Frame(MainWin, height=300, width=600)
# 设定四个框架的网格的行列，“田”字型，是两行两列，行：0~1，列是0~1
InputFrame.grid(row=0, column=0)
OutputFrame.grid(row=0, column=1)
butFrame.grid(row=1, column=0)
DataFrame.grid(row=1, column=1, sticky=tk.N)
# InputFrame框架，首行放 一个文字标签 “识别结果：" 和一个输出框Entry
# 第二行是logo 图片窗口
ResultStr = tk.StringVar()  # 设置一个结果显示的字符串变量和输出结果相关
Lab1 = tk.Label(InputFrame, text='运行结果:', font=('Arial', 12), width=20, height=1)
Lab1.grid(row=0, column=0, padx=10, pady=20)
entry1 = tk.Entry(InputFrame, font=('Arial', 12), width=20, textvariable=ResultStr)
entry1.grid(row=0, column=1)
ResultStr.set('结果字符串')
LogoImage = tk.PhotoImage(file='ImageFile/KmeansLogo.png')
Lab2 = tk.Label(InputFrame, image=LogoImage)
Lab2.grid(row=1, column=0, columnspan=2, padx=40)
# OutputFrame 首行放 一个文字提示标签 “训练误差曲线"
# 第二行是一个显示曲线的画布窗口
Lab3 = tk.Label(OutputFrame, text='图像聚类结果', font=('Arial', 14), width=16, height=1)
Lab3.grid(row=0, column=0, pady=20)
fig = plt.Figure(figsize=(5, 4), dpi=100)  # 设置空画布窗口，figsize为大小（英寸），dpi为分辨率
#设置输出显示画布
draw_set = FigureCanvasTkAgg(fig, master=OutputFrame)  # 将空画布设置在tkinter的输出容器OutputFrame上
ax1 = fig.add_subplot(2, 2, 1) #画布中的第1个子窗口
ax2 = fig.add_subplot(2, 2, 2) #画布中的第2个子窗口
ax3 = fig.add_subplot(2, 2, 3) #画布中的第3个子窗口
ax4 = fig.add_subplot(2, 2, 4) #画布中的第4个子窗口
ax1.set_axis_off()#关掉坐标轴
ax2.set_axis_off()
ax3.set_axis_off()
ax4.set_axis_off()
draw_set.get_tk_widget().grid(row=1, column=0)  # ,height=460,width=550)
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16, }  # 设置字体的字典
#ax1.set_xlabel('x-age', font2)  # x轴名称并附带字体
#ax1.set_ylabel('y-dep', font2)  # y轴名称并附带字体
# ButtonFrame 第1行是是10个单选无线按钮
Target = [('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9)]
TargetV = tk.IntVar()
Target_startx = 50
Target_starty = 20
# 为简化，用一个循环画出10个单选按钮，注意它们公用一个响应函数SetKvalue
for txt, num in Target:
    rbut = tk.Radiobutton(butFrame, text=txt, value=num, font=('Arial', 12), width=3, height=1, command=SetKvalue,
                          variable=TargetV)
    rbut.place(x=Target_startx + num * 50, y=Target_starty)
# 第二行就是所有功能按钮放置区，这里有9个，具体功能如下
but1 = tk.Button(butFrame, text='装入图片', font=('Arial', 12), width=10, height=1, command=LoadImage)
but2 = tk.Button(butFrame, text='添加样本', font=('Arial', 12), width=10, height=1, command=callback)
but3 = tk.Button(butFrame, text='保存样本', font=('Arial', 12), width=10, height=1, command=callback)
but4 = tk.Button(butFrame, text='装入样本', font=('Arial', 12), width=10, height=1, command=LoadSample)
but5 = tk.Button(butFrame, text='K-均值测试', font=('Arial', 12), width=10, height=1, command=CallKmeans)
but6 = tk.Button(butFrame, text='装入iris', font=('Arial', 12), width=10, height=1, command=LoadIris)
but7 = tk.Button(butFrame, text='装物流位置', font=('Arial', 12), width=10, height=1, command=LoadCoordinate)
but8 = tk.Button(butFrame, text='装客户信息', font=('Arial', 12), width=10, height=1, command=LoadCustomerInfo)
but9 = tk.Button(butFrame, text='删除样本', font=('Arial', 12), width=10, height=1, command=callback)
# 下面是按钮放置的位置
but1.place(x=50, y=80)
but2.place(x=250, y=80)
but3.place(x=450, y=80)
but4.place(x=50, y=120)
but5.place(x=250, y=120)
but6.place(x=450, y=120)
but7.place(x=50, y=160)
but8.place(x=250, y=160)
but9.place(x=450, y=160)
but1.config(state=tk.ACTIVE)
but2.config(state=tk.DISABLED)
but3.config(state=tk.DISABLED)
but4.config(state=tk.DISABLED)
but5.config(state=tk.DISABLED)
but6.config(state=tk.DISABLED)
but7.config(state=tk.DISABLED)
but8.config(state=tk.DISABLED)
but9.config(state=tk.DISABLED)
# DataFrame 数据输出框，主要放置输出的数据和它们的标签
CountStr = tk.StringVar()  # 对应的样本计数
EpochStr = tk.StringVar()  # 循环计数
ValueStr = tk.StringVar()  # 最后的训练误差
Lab4 = tk.Label(DataFrame, text='训练样本:', font=('Arial', 12), width=10, height=1)
Lab4.grid(row=0, column=0, pady=20)
Lab5 = tk.Label(DataFrame, text='聚类数K:', font=('Arial', 12), width=10, height=1)
Lab5.grid(row=1, column=0)
Lab6 = tk.Label(DataFrame, text='训练比例:', font=('Arial', 12), width=10, height=1)
Lab6.grid(row=2, column=0, pady=20)
entry4 = tk.Entry(DataFrame, font=('Arial', 12), width=15, textvariable=CountStr)
entry4.grid(row=0, column=1, pady=20)
CountStr.set("0")
entry5 = tk.Entry(DataFrame, font=('Arial', 12), width=15, textvariable=EpochStr)
entry5.grid(row=1, column=1)
EpochStr.set("0")
entry6 = tk.Entry(DataFrame, font=('Arial', 12), width=15, textvariable=ValueStr)
entry6.grid(row=2, column=1, pady=20)
#ErrStr.set("0")
# 建立主循环，准备接收消息并相应
MainWin.mainloop()

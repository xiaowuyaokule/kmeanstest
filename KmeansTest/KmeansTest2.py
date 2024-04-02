'''
这是一个K-均值聚类算法的GUI界面下的演示程序，演示15个二维数据点的聚类过程
'''
import cv2
import pandas as pd
import time
from PIL import ImageTk, Image
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tkinter as tk
import tkinter.filedialog as file
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # 导入在tkinter 内嵌入的画布子窗口

# from matplotlib.figure import Figure
# GUI 初始化
MainWin = tk.Tk()  # 建立主窗口
MainWin.title('机器学习K-均值算法演示')  # 窗口标题
MainWin.geometry('1200x700')  # 窗口宽X高尺寸
MainWin.resizable(width=True, height=True)  # 设定窗口尺寸可变
TagFlag = False
FinishFlag = False
# KNN 算法演示程序-鸢尾花分类

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
    ax.set_autoscale_on(True)  # 设置自动量程
    ax.scatter(X[:, 0], X[:, 1], c="b")
    draw_set.draw()
    # 随机初始化中心点
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for N in range(max_iters):
        # 计算每个样本到中心点的距离
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        # print(distances)
        # 分配样本到最近的中心点所在的簇
        labels = np.argmin(distances, axis=1)
        # print("labels",labels)
        # 更新中心点为每个簇的均值
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        # 如果中心点没有变化，提前结束迭代
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        # print(centroids)
        # 显示聚类的过程
        ax.clear()
        for i in range(len(X)):
            ax.plot([centroids[labels[i], 0], X[i, 0]], [centroids[labels[i], 1], X[i, 1]], c="r", marker="o")
        ax.scatter(centroids[:, 0], centroids[:, 1], c="b", linewidth=7)
        # draw_set.draw()
        # x_data = centroids[:, 0]
        # y_data = centroids[:, 1]
        # MainWin.update()
        print("\nN=", N)
    return centroids, labels


# K-均值响应函数
# 鸢尾花的训练样本及标签，测试样本及标签
def CallKmeans():
    global data, K
    centroids, labels = Kmeans(data, K)
    fileName = 'ImageFile/KmeansLogo.jpg'
    print("center=\n", centroids, "\nlabel=", labels)
    ax.scatter(centroids[:, 0], centroids[:, 1], c="g")
    draw_set.draw()
    picshow(fileName)
    ResultStr.set("聚类完成")
    MainWin.mainloop()
    return


def callback():
    print("被调用了")
    return


# 显示一幅图像的函数
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


def SetKvalue():
    global K
    k = TargetV.get()
    if 1 < k < 10:
        K = k
        EpochStr.set(str(K))


data = []
K = 3


def LoadIris():
    # data1 = pd.read_excel('iris.xlsx', sheet_name='iris')  # 训练数据集
    # print(data1)
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
Lab3 = tk.Label(OutputFrame, text='聚类结果', font=('Arial', 14), width=10, height=1)
Lab3.grid(row=0, column=0, pady=20)
fig = plt.Figure(figsize=(5, 4), dpi=100)  # 设置空画布窗口，figsize为大小（英寸），dpi为分辨率

draw_set = FigureCanvasTkAgg(fig, master=OutputFrame)  # 将空画布设置在tkinter的输出容器OutputFrame上
ax = fig.add_subplot(1, 1, 1)  # 在plt窗口设置坐标系ax

draw_set.get_tk_widget().grid(row=1, column=0)  # ,height=460,width=550)
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16, }  # 设置字体的字典
ax.set_xlabel('K-value', font2)  # x轴名称并附带字体
ax.set_ylabel('ResultErr', font2)  # y轴名称并附带字体
# ButtonFrame 第1行是是10个单选无线按钮
Target = [('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9)]
TargetVector = np.zeros(10)
TargetV = tk.IntVar()
Target_startx = 50
Target_starty = 20
# 为简化，用一个循环画出10个单选按钮，注意它们公用一个响应函数SetTarget
for txt, num in Target:
    rbut = tk.Radiobutton(butFrame, text=txt, value=num, font=('Arial', 12), width=3, height=1, command=SetKvalue,
                          variable=TargetV)
    rbut.place(x=Target_startx + num * 50, y=Target_starty)
# 第二行就是所有功能按钮放置区，这里有9个，具体功能如下
but1 = tk.Button(butFrame, text='选择图片', font=('Arial', 12), width=10, height=1, command=callback)
but2 = tk.Button(butFrame, text='添加样本', font=('Arial', 12), width=10, height=1, command=callback)
but3 = tk.Button(butFrame, text='保存样本', font=('Arial', 12), width=10, height=1, command=callback)
but4 = tk.Button(butFrame, text='装入样本', font=('Arial', 12), width=10, height=1, command=LoadSample)
but5 = tk.Button(butFrame, text='K-均值测试', font=('Arial', 12), width=10, height=1, command=CallKmeans)
but6 = tk.Button(butFrame, text='装入iris', font=('Arial', 12), width=10, height=1, command=LoadIris)
but7 = tk.Button(butFrame, text='保存网络', font=('Arial', 12), width=10, height=1, command=callback)
but8 = tk.Button(butFrame, text='装入网络', font=('Arial', 12), width=10, height=1, command=callback)
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
but1.config(state=tk.DISABLED)
but2.config(state=tk.DISABLED)
but3.config(state=tk.DISABLED)
but4.config(state=tk.ACTIVE)
but5.config(state=tk.DISABLED)
but6.config(state=tk.DISABLED)
but7.config(state=tk.DISABLED)
but8.config(state=tk.DISABLED)
but9.config(state=tk.DISABLED)
# DataFrame 数据输出框，主要放置输出的数据和它们的标签
CountStr = tk.StringVar()  # 对应的样本计数
EpochStr = tk.StringVar()  # 循环计数
ErrStr = tk.StringVar()  # 最后的训练误差
Lab4 = tk.Label(DataFrame, text='训练样本:', font=('Arial', 12), width=10, height=1)
Lab4.grid(row=0, column=0, pady=20)
Lab5 = tk.Label(DataFrame, text='聚类数K:', font=('Arial', 12), width=10, height=1)
Lab5.grid(row=1, column=0)
Lab6 = tk.Label(DataFrame, text='训练误差:', font=('Arial', 12), width=10, height=1)
Lab6.grid(row=2, column=0, pady=20)
entry4 = tk.Entry(DataFrame, font=('Arial', 12), width=15, textvariable=CountStr)
entry4.grid(row=0, column=1, pady=20)
CountStr.set("0")
entry5 = tk.Entry(DataFrame, font=('Arial', 12), width=15, textvariable=EpochStr)
entry5.grid(row=1, column=1)
EpochStr.set("0")
entry6 = tk.Entry(DataFrame, font=('Arial', 12), width=15, textvariable=ErrStr)
entry6.grid(row=2, column=1, pady=20)
ErrStr.set("0")
# 建立主循环，准备接收消息并相应
MainWin.mainloop()

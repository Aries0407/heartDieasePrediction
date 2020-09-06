import numpy as np
import pandas as pd
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=RuntimeWarning)
# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=RuntimeWarning)

'''
age 年龄
sex 性别 1=male,0=female
cp  胸痛类型(4种) 值1:典型心绞痛，值2:非典型心绞痛，值3:非心绞痛，值4:无症状
trestbps 静息血压 
chol 血清胆固醇
fbs 空腹血糖 >120mg/dl ,1=true; 0=false
restecg 静息心电图(值0,1,2)
thalach 达到的最大心率
exang 运动诱发的心绞痛(1=yes;0=no)
oldpeak 相对于休息的运动引起的ST值(ST值与心电图上的位置有关)
slope 运动高峰ST段的坡度 Value 1: upsloping向上倾斜, Value 2: flat持平, Value 3: downsloping向下倾斜
ca  The number of major vessels(血管) (0-3)
thal A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
       一种叫做地中海贫血的血液疾病(3 =正常;6 =固定缺陷;7 =可逆转缺陷)
target 生病没有(0=no,1=yes)
'''

#sigmoid函数
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def classify(z):
    if sigmoid(z)>0.5:
        return 1
    else:
        return 0

def lr_train_bgd(feature,label,maxCycle,alpha):
    '''
    利用随机梯度下降法训练模型
    input：
    :param feature:feature(mat)：特征
    :param label:label(mat)：标签
    :param maxCycle:maxCycle(int)：最大迭代次数
    :param alpha:alpha(float)：学习率
    output：
    w（mat）：权重
    :return:
    '''
    n = np.shape(feature)[1]
    w = np.mat(np.ones((n,1)))
    i=0
    while i<= maxCycle:
        i+=1
        h = sigmoid(feature*w)
        err = label - h
        if i%100==0:
            print("iter="+str(i) + ", train error rate = " + str(error_rate(h,label)))
        w = w + alpha *feature.T * err
    return w

def error_rate(h,label):
    '''
    计算当前损失函数的值
    input:
    :param h:h(mat)：预测值
    :param label:label(mat):实际值
    output：
    err/m(float) :错误率
    :return:
    '''
    m = np.shape(h)[0]
    sum_err = 0
    for i in range(m):
        if h[i,0]> 0 and (1-h[i,0]) > 0:
            sum_err -= (label[i,0] * np.log(h[i,0]) + 1- label[i,0] * np.log(1-h[i,0]))
        else:
            sum_err-=0
    return sum_err/m

def load_data(file_name):
    # 解决matplotlib中文问题，没看懂！！！！！
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # 导入数据
    df = pd.read_csv('heart_disease_data/heart.csv')
    # 查看总体数据情况
    # df.info()
    '''
    df.describe()
    df.target.value_counts()
    '''
    sns.countplot(x='target', data=df, palette="muted")
    plt.xlabel("得病/未得病比例")
    # Text(0.5,0,'Sex (0 = 女, 1= 男)')
    plt.figure(figsize=(18, 7))
    sns.countplot(x='age', data=df, hue='target', palette='PuBuGn', saturation=0.8)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()
    f = open(file_name)
    datas = f.readlines()[1::]
    random.shuffle(datas)
    n = len(datas) // 3
    test = datas[0:n]
    train = datas[n::]
    feature_data = []
    label_data = []
    test_set = []

    for line in train:
        feature_temp = []
        label_temp = []
        lines = line.strip().split(",")
        feature_temp.append(1)  # 为特征添加偏置项，添加到了列表首位

        for i in range(len(lines) - 1):
            feature_temp.append(float(lines[i]))
        label_temp.append(float(lines[-1]))

        feature_data.append(feature_temp)
        label_data.append(label_temp)
    for line in test:
        test_set_row = []
        lines = line.strip().split(",")
        test_set_row.append(1)  # 为特征添加偏置项，添加到了列表首位
        for i in range(len(lines)):
            test_set_row.append(float(lines[i]))
        test_set.append(test_set_row)
    f.close()
    return np.mat(feature_data),np.mat(label_data),test_set

def save_model(file_name,w):
    '''
    保存最终的模型
    input:
    :param file_name(string):模型的文件名
    :param w(mat):LR模型的权重
    '''
    m = np.shape(w)[0]
    f_w = open(file_name,'w')
    w_array = []
    for i in range(m):
        w_array.append(str(w[i,0]))
    f_w.write("\t".join((w_array)))
    f_w.close()

if __name__ == '__main__':
   feature,label,test = load_data('heart_disease_data\\heart.txt')
   w = lr_train_bgd(feature,label,100000,0.01)
   save_model('heart_disease_data\\model.txt',w)
   f = open('heart_disease_data\\model.txt')
   w = f.readline()
   w = w.split("\t")
   weight = []
   for i in w:
       weight.append(float(i))

   ##模型评估阶段
   # 测试阶段并绘制混淆矩阵
   FACT = []
   PRED = []
   TP = FP = FN = TN = 0
   total = len(test)

   '''
   result 表示预测值，i[14]表示实际值
   '''
   for i in test:
       z = 0
       for j in range(len(weight)):
           x = i[j]
           x = float(x)
           z += (float(weight[j])*float(x))
       z = round(z,1)
       predict = classify(z)
       PRED.append(predict)
       fact = int(i[len(i)-1])
       FACT.append(fact)
       if fact == 1 and predict == 1:
           TP += 1
       if fact == 0 and predict == 0:
           TN += 1
       if fact == 0 and predict == 1:
           FP += 1
       if fact == 1 and predict == 0:
           FN += 1

   print("准确率：{:.2f}%".format(100 * (TP + TN) / total))
   print("精确率：{:.2f}%".format(100 * TP / (TP + FP)))
   print("(真阳率)召回率：{:.2f}%".format(100 * TP / (TP + FN)))
   print("假阳率：{:.2f}%".format(100 * FP / (FP + TN)))

   classes = list(set(FACT))
   classes.sort()
   confusion = confusion_matrix(PRED, FACT)
   plt.imshow(confusion, cmap=plt.cm.Blues)
   indices = range(len(confusion))
   plt.xticks(indices, classes)
   plt.yticks(indices, classes)
   # ticks 这个是坐标轴上的坐标点
   # label 这个是坐标轴的注释说明
   plt.colorbar()
   plt.xlabel('PRED')
   plt.ylabel('FACT')
   for first_index in range(len(confusion)):
       for second_index in range(len(confusion[first_index])):
           plt.text(first_index, second_index, confusion[first_index][second_index])
   plt.show()

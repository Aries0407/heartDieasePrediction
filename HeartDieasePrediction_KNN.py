import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 解决matplotlib中文问题，没看懂！！！！！
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 导入数据
df = pd.read_csv('heart_disease_data/heart.csv')
#查看总体数据情况
#df.info()
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

'''
df.describe()
df.target.value_counts()
'''

sns.countplot(x='target',data=df,palette="muted")
plt.xlabel("得病/未得病比例")
# Text(0.5,0,'Sex (0 = 女, 1= 男)')
plt.figure(figsize=(18,7))
sns.countplot(x='age',data = df, hue = 'target',palette='PuBuGn',saturation=0.8)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()

# 数据读取
with open('heart_disease_data\\heart.txt','r',encoding="gbk") as file:
    reader = csv.DictReader(file)
    datas = [row for row in reader]

#分组
#随机打乱数据顺序
random.shuffle(datas)
n = len(datas)//3
test_set = datas[0:n]
train_set = datas[n:]

#基于KNN算法来实现心脏病的预测

#求欧几里得距离
def distance(d1,d2):
    res = 0
    for key in ("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"):
        res += (float(d1[key])-float(d2[key]))**2
    return res**0.5

K=5
def knn(data):
    #1、求距离
    res = [
        {"result":train["target"],"distance":distance(data,train)}
        for train in train_set
    ]
    #2、升序排列
    res = sorted(res, key=lambda item: item["distance"])
    #3、取前K个
    res2 = res[0:K]
    #4、加权平均
    result = {"0": 0 ,"1": 1}
    #总距离
    sum = 0
    for r in res2:
        sum+=r["distance"]

    for r in res2:
       result[r['result']]+=1-r['distance']/sum

    if result["0"]>result["1"]:
        return "0"
    else:
        return "1"

#测试阶段并绘制混淆矩阵
FACT = []
PRED = []
TP=FP=FN=TN=0
total = len(test_set)
for test in test_set:
    '''
    fact代表真实值，predict代表预测值
    '''
    fact = test['target']
    FACT.append(int(fact))
    predict = knn(test)
    PRED.append(int(predict))
    if fact == '1' and predict == '1':
       TP+=1
    if fact == '0' and predict == '0':
       TN+=1
    if fact == '0' and predict == '1':
       FP+=1
    if fact =='1' and predict =='0':
       FN+=1


print("准确率：{:.2f}%".format(100*(TP+TN)/total))
print("精确率：{:.2f}%".format(100*TP/(TP+FP)))
print("(真阳率)召回率：{:.2f}%".format(100*TP/(TP+FN)))
print("假阳率：{:.2f}%".format(100*FP/(FP+TN)))

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

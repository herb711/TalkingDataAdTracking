
'''
ip: ip address of click.
app: app id for marketing.
device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
os: os version id of user mobile phone
channel: channel id of mobile ad publisher
click_time: timestamp of click (UTC)
attributed_time: if user download the app for after clicking an ad, this is the time of the app download
is_attributed: the target that is to be predicted, indicating the app was downloaded

'''

#### Libraries
import pandas as pd 
import numpy as np
import sklearn.preprocessing as preprocessing
import csv  
 

def load_data(s):# csv数据读取
    if s == 0:
        df = pd.read_csv('TalkingDataAdTracking/data/train_sample.csv') #读取训练数据
    else:
        df = pd.read_csv('TalkingDataAdTracking/data/test.csv') #读取竞赛数据

    #时间转换
    df['click_time'] = pd.to_datetime(df['click_time'])
    day = [i.day for i in df['click_time']]
    hour = [i.hour for i in df['click_time']]
    minute = [i.minute for i in df['click_time']]
    time = pd.DataFrame({'day':day,'hour':hour,'minute':minute})   
    df = pd.concat([df, time], axis=1) #加入到数据中

    # 用preprocessing模块做scaling
    scaler = preprocessing.StandardScaler()
    #df.fillna(0.0, inplace = True)  #填充所有缺失数据
    #df.ip = df.ip.fillna(value=0.0)
    scale_param = scaler.fit(df['ip'].values.reshape(-1, 1))
    df['ip_scaled'] = scaler.fit_transform(df['ip'].values.reshape(-1, 1), scale_param)

    scale_param = scaler.fit(df['app'].values.reshape(-1,1))
    df['app_scaled'] = scaler.fit_transform(df['app'].values.reshape(-1,1), scale_param) 

    scale_param = scaler.fit(df['device'].values.reshape(-1,1))
    df['device_scaled'] = scaler.fit_transform(df['device'].values.reshape(-1,1), scale_param) 

    scale_param = scaler.fit(df['os'].values.reshape(-1,1))
    df['os_scaled'] = scaler.fit_transform(df['os'].values.reshape(-1,1), scale_param) 

    scale_param = scaler.fit(df['channel'].values.reshape(-1,1))
    df['channel_scaled'] = scaler.fit_transform(df['channel'].values.reshape(-1,1), scale_param) 

    scale_param = scaler.fit(df['day'].values.reshape(-1,1))
    df['day_scaled'] = scaler.fit_transform(df['day'].values.reshape(-1,1), scale_param) 

    scale_param = scaler.fit(df['hour'].values.reshape(-1,1))
    df['hour_scaled'] = scaler.fit_transform(df['hour'].values.reshape(-1,1), scale_param) 

    scale_param = scaler.fit(df['minute'].values.reshape(-1,1))
    df['minute_scaled'] = scaler.fit_transform(df['minute'].values.reshape(-1,1), scale_param) 

    return df

def load_data_wrapper():

    #读取训练数据
    train_data = load_data(0)
    #将特征值取出并转换为矩阵
    train_df = train_data.filter(regex='is_attributed|ip_scaled|app_scaled|device_scaled|os_scaled|channel_scaled|day_scaled|hour_scaled|minute_scaled')
    #print(train_df.columns) # 打印列索引
    train_np = train_df.as_matrix()
    # print(train_df.dtypes) #查看不同列的数据类型
    # y即Survival结果
    trd_y = train_np[:, 0]
    # X即特征属性值
    trd_X = train_np[:, 1:]

    #读取竞赛数据
    test_data = load_data(0)
    #将特征值取出
    test_df = test_data.filter(regex='is_attributed|ip_scaled|app_scaled|device_scaled|os_scaled|channel_scaled|day_scaled|hour_scaled|minute_scaled')
    test_np = test_df.as_matrix()
    # y即Id
    test_y = test_data['is_attributed'].as_matrix()
    # X即特征属性值
    test_X = test_np

    return trd_X, trd_y, test_X, test_y

def save_data(X, y):
    result = pd.DataFrame({'click_id':y.astype(np.int), 'is_attributed':X.astype(np.int32)})
    result.to_csv("Titanic/data/result.csv", index=False)

def save_error(X):
    with open('Titanic/data/error.csv',"w") as csvfile: 
        writer = csv.writer(csvfile)
        #先写入columns_name
        #writer.writerow(["PassengerId","Survived"])
        #写入多行用writerows
        writer.writerows(X)


load_data_wrapper()


def save_data0(X, y):
    with open('Titanic/data/result.csv',"w") as csvfile: 
        writer = csv.writer(csvfile)
        #先写入columns_name
        writer.writerow(["PassengerId","Survived"])
        #写入多行用writerows
        writer.writerows(X, y)
        

'''
查找出错项
origin_data_train = pd.read_csv("/Users/HanXiaoyang/Titanic_data/Train.csv")
bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
bad_cases

'''
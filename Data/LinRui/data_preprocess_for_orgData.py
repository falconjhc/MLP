import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import numpy as np

dataset_model = pd.read_csv('./Data/LinRui/orgData.csv', sep=',')
dataset_model = dataset_model.fillna(0)
dim_size = dataset_model.columns.size
sc = StandardScaler()
ros = RandomOverSampler()

valid_data = dataset_model.iloc[:,2:dim_size-1]
y = dataset_model.iloc[:,dim_size-1]
x = pd.get_dummies(valid_data,drop_first=True)
x = x.values
x = sc.fit_transform(x)

train_data, test_data, train_label, test_label = train_test_split(x,y,test_size=0.2, random_state=0)
train_data, train_label = ros.fit_sample(train_data, train_label)
test_label = test_label.values

train_data_save = np.concatenate([train_data,np.reshape(train_label,[train_label.shape[0],1])],axis=1)
test_data_save = np.concatenate([test_data,np.reshape(test_label,[test_label.shape[0],1])],axis=1)

column_name = list()
for ii in range(train_data_save.shape[1]):
    column_name.append(str(ii))
train_data_save = pd.DataFrame(train_data_save,
                               columns=column_name)
test_data_save = pd.DataFrame(test_data_save,
                              columns=column_name)

train_data_save.to_csv('./Data/LinRui/TrainData.csv',sep=',',index_label='-1')
test_data_save.to_csv('./Data/LinRui/TestData.csv',sep=',',index_label='-1')


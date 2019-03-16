import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA




discrete_dim=[3]
dataset_model = pd.read_csv('../Data/data_ML_Exit1.csv', sep=',')
keep_componenent=75


def get_dimension_data(full_x,selected_dim):
    for ii in selected_dim:
        if ii == selected_dim[0]:
            output_pattern = np.reshape(full_x[:, ii], newshape=[full_x.shape[0], 1])
        else:
            output_pattern = np.concatenate([output_pattern,
                                             np.reshape(full_x[:, ii],
                                                        newshape=[full_x.shape[0], 1])],
                                            axis=1)
    return output_pattern

def to_onehot(input_pattern):
    def imp(input_label, involved_label_list):

        # for abnormal label process:
        # for those labels in the test list not occurring in the training
        batch_size=input_label.shape[0]
        abnormal_marker_indices = list()
        for ii in range(len(input_label)):
            if not input_label[ii] in involved_label_list:
                abnormal_marker_indices.append(ii)
        data_indices = np.arange(len(input_label))
        data_indices_list = data_indices.tolist()
        for ii in abnormal_marker_indices:
            data_indices_list.remove(data_indices[ii])
        data_indices = np.array(data_indices_list)

        label_length = len(involved_label_list)
        input_label_matrix = np.tile(np.asarray(input_label), [len(involved_label_list), 1])
        fine_tune_martix = np.transpose(np.tile(involved_label_list, [batch_size, 1]))

        diff = input_label_matrix - fine_tune_martix
        find_positions = np.argwhere(np.transpose(diff) == 0)
        input_label_indices = np.transpose(find_positions[:, 1:]).tolist()

        output_one_hot_label = np.zeros((len(input_label), label_length), dtype=np.float32)
        if data_indices.tolist():
            output_one_hot_label[data_indices, input_label_indices] = 1
        return output_one_hot_label


    pattern_dim = input_pattern.shape[1]
    for ii in range(pattern_dim):
        current_data = input_pattern[:,ii]
        data_vec=np.unique(current_data)
        new_current_data = -np.ones(shape=current_data.shape,
                                     dtype=np.float32)
        for counter in range(len(data_vec)):
            new_current_data[np.where(current_data==data_vec[counter])]=counter
        new_data_vec = np.unique(new_current_data)

        onehot = imp(input_label=new_current_data,
                     involved_label_list=new_data_vec)
        if ii == 0:
            full_onehot = onehot
        else:
            full_onehot = np.concatenate([full_onehot,onehot],axis=1)

    return full_onehot

def to_number(input_pattern):
    pattern_dim = input_pattern.shape[1]
    for ii in range(pattern_dim):
        current_data = input_pattern[:, ii]
        data_vec = np.unique(current_data)
        data_vec.sort()
        number=np.ones(dtype=np.float32,shape=current_data.shape)
        counter=0
        for jj in data_vec:
            number[np.where(current_data==jj)]=counter
            counter+=1
        if ii == 0:
            output_number = np.reshape(number,newshape=[number.shape[0],1])
        else:
            output_number = np.concatenate([output_number,
                                            np.reshape(number,newshape=[number.shape[0],1])],axis=1)


    return output_number


dataset_model = dataset_model.iloc[1:,:]
dataset_model = dataset_model.fillna(0)
dataset_value = dataset_model.values
dim_size = dataset_value.shape[1]

full_y = dataset_value[:,0]
full_x = dataset_value[:,1:]

full_dim=range(0,dim_size-1)
continuous_dim = [ii for ii in full_dim if ii not in discrete_dim]



discrete_pattern = get_dimension_data(full_x=full_x,
                                      selected_dim=discrete_dim)
continuous_pattern = get_dimension_data(full_x=full_x,
                                        selected_dim=continuous_dim)


sc = StandardScaler()
ros = RandomOverSampler()

continuous_pattern_scaled = sc.fit_transform(continuous_pattern)
discrete_pattern_onehot = to_onehot(input_pattern=discrete_pattern)
pattern = np.concatenate([continuous_pattern_scaled,
                          discrete_pattern_onehot], axis=1)






train_data, test_data, train_label, test_label = train_test_split(pattern,full_y,test_size=0.2, random_state=0)

# svd = TruncatedSVD(n_components=keep_componenent,algorithm='arpack')
# coef = svd.fit(train_data)
# train_data=svd.transform(train_data)
# test_data=svd.transform(test_data)

# pca = PCA(n_components=keep_componenent)
# coef = pca.fit(train_data)
# train_data = pca.transform(train_data)
# test_data = pca.transform(test_data)

# pca = SparsePCA(n_components=keep_componenent)
# coef = pca.fit(train_data)
# train_data = pca.transform(train_data)
# test_data = pca.transform(test_data)


train_data, train_label = ros.fit_sample(train_data, train_label)


train_data_save = np.concatenate([train_data,np.reshape(train_label,[train_label.shape[0],1])],axis=1)
test_data_save = np.concatenate([test_data,np.reshape(test_label,[test_label.shape[0],1])],axis=1)

column_name = list()
for ii in range(train_data_save.shape[1]):
    column_name.append(str(ii))
train_data_save = pd.DataFrame(train_data_save,
                               columns=column_name)
test_data_save = pd.DataFrame(test_data_save,
                              columns=column_name)

train_data_save.to_csv('./data_ML_Exit1_Train.csv',sep=',',index_label='-1')
test_data_save.to_csv('./data_ML_Exit1_Test.csv',sep=',',index_label='-1')


clear all;
data_org=load('./DataVersion02/OriginalData/data_org.mat');
data_org=data_org.data_org;
continous_feautre_ids=[7,8,21,22];



probability_0_indices = find(data_org(:,1)==0);
probability_1_indices = find(data_org(:,1)==1);


selected_0_indices = (randperm(length(probability_0_indices),int32(length(probability_0_indices)*0.25)))';
selected_1_indices = (randperm(length(probability_1_indices),int32(length(probability_1_indices)*0.25)))';



selected_0_indices_test = probability_0_indices(selected_0_indices);
selected_1_indices_test = probability_1_indices(selected_1_indices);

selected_0_indices_train = setdiff(probability_0_indices,selected_0_indices_test);
selected_1_indices_train = setdiff(probability_1_indices,selected_1_indices_test);

expand_rate = ceil(length(selected_0_indices_train) / length(selected_1_indices_train))-1;
for ii=1:expand_rate
    if ii == 1
        output_selected_1_indices_train = selected_1_indices_train;
    else
        output_selected_1_indices_train = [output_selected_1_indices_train;selected_1_indices_train];
    end
end


data_train = [data_org(selected_0_indices_train,:); data_org(output_selected_1_indices_train,:)];
data_test = [data_org(selected_0_indices_test,:); data_org(selected_1_indices_test,:)];


data_train_values = data_train(:,1);
data_test_values = data_test(:,1);
data_train_features = data_train(:,2:end);
data_test_features = data_test(:,2:end);


[data_train_features, data_test_features] = DiscreteFeatureCheck(data_train_features, data_test_features,continous_feautre_ids);
[data_train_features, data_test_features] = FeatureNormalization(data_train_features, data_test_features,continous_feautre_ids);


data_train=[data_train_values,data_train_features];
data_test=[data_test_values,data_test_features];
save('./DataVersion02/divide1_train.mat','data_train')
save('./DataVersion02/divide1_test.mat','data_test')


csvwrite('./DataVersion02/divide1_train.csv',data_train);
csvwrite('./DataVersion02/divide1_test.csv',data_test);




function [ output_feature_train, output_feature_test ] = DiscreteFeatureCheck( input_feature_train,input_feature_test,continus_feature_id)
feature_dimension = size(input_feature_train,2);
output_feature_train=input_feature_train;
output_feature_test=input_feature_test;


for ii=1:feature_dimension
    
    if find(continus_feature_id==ii)
        continue
    end
    
    current_dimension_train = output_feature_train(:,ii);
    current_dimension_test = output_feature_test(:,ii);
    feature_unique_train=unique(current_dimension_train);
    feature_unique_test=unique(current_dimension_test);
    
    for jj=1:length(feature_unique_test)
        current_test_feature=feature_unique_test(jj);
        if isempty(find(feature_unique_train==current_test_feature))==1 % didnt find the current_test_feature on the train features`c
            
            % delievery data from test to train
            relevant_test_data_indices=find(current_dimension_test==current_test_feature);
            relevant_test_data=output_feature_test(relevant_test_data_indices,:);
            output_feature_test(relevant_test_data_indices,:)=[];
            
            
            selected_delievery_data_num = int32(length(size(relevant_test_data,1))*0.75);
            selected_test_data_to_train_indices=randperm(size(relevant_test_data,1), selected_delievery_data_num);
            non_selected_test_data_to_train_indices=setdiff(1:size(relevant_test_data,1),selected_test_data_to_train_indices);
            output_feature_train=[output_feature_train;relevant_test_data(selected_test_data_to_train_indices,:)];
            output_feature_test=[output_feature_test;relevant_test_data(non_selected_test_data_to_train_indices,:)];
            
            
            % delivery data from train back to test
            valid=false;
            while (~valid)
                updated_current_dimension_train=output_feature_train(:,ii);
                updated_feature_unique_train=unique(updated_current_dimension_train);
                selected_train_data_to_test_indices=randperm(size(output_feature_train,1), selected_delievery_data_num);
                non_selected_train_data_to_test_indices=setdiff(1:size(output_feature_train,1),selected_train_data_to_test_indices);
                non_selected_train_data_to_test_unique=unique(updated_current_dimension_train(non_selected_train_data_to_test_indices,:));
                if length(non_selected_train_data_to_test_unique)==length(updated_feature_unique_train)
                    valid=true;
                end
            end
            
            
            selected_train_data=output_feature_train(selected_train_data_to_test_indices,:);
            output_feature_train(selected_train_data_to_test_indices,:)=[];
            output_feature_test=[output_feature_test;selected_train_data];
            
            
            
            current_dimension_test = output_feature_test(:,ii);
            
        end
    end
    
end

end


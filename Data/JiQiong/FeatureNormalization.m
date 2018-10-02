function [ output_feature_train, output_feature_test ] = FeatureNormalization(input_features_train, input_features_test, continus_feature_id)


feature_dimension = size(input_features_train,2);
output_feature_train=input_features_train;
output_feature_test=input_features_test;
for ii=1:feature_dimension
    current_dimension_train = input_features_train(:,ii);
    current_dimension_test = input_features_test(:,ii);
    if isempty(find(continus_feature_id==ii))~=1
        [current_dimension_train,current_dimension_test] = continus_feature(current_dimension_train,current_dimension_test);
        output_feature_train(:,ii)=current_dimension_train;
        output_feature_test(:,ii)=current_dimension_test;
    else
        [current_dimension_train,current_dimension_test] = discrete_feature(current_dimension_train,current_dimension_test);
        output_feature_train(:,ii)=current_dimension_train;
        output_feature_test(:,ii)=current_dimension_test;
    end
end



end

function [ output_train, output_test ] = discrete_feature(input_train,input_test)
output_train=input_train;
output_test=input_test;
feature_unique_train=unique(input_train);
feature_unique_test=unique(input_test);

valid_train_features=true;
for ii=1:length(feature_unique_test)
    current_test_feature=feature_unique_test(ii);
    if isempty(find(feature_unique_train==current_test_feature))==1
        disp(sprintf('ERROR!!!'));
        valid_train_features=false;
        break;
    else
        
    end
    
end

if valid_train_features
    for ii=1:length(feature_unique_train)
        current_train_feature=feature_unique_train(ii);
        indices_in_train=find(input_train==current_train_feature);
        indices_in_test=find(input_test==current_train_feature);
        output_train(indices_in_train)=ii-1;
        output_test(indices_in_test)=ii-1;
    end
else
    disp(sprintf('ERROR!!!'));
    return
    
end

end



function [ output_train, output_test ] = continus_feature(input_train,input_test)
output_train=input_train;
output_test=input_test;
minv=min(output_train);
output_train=output_train-minv;
output_test=output_test-minv;
maxv=max(output_train);
output_train=output_train/maxv;
output_test=output_test/maxv;
end
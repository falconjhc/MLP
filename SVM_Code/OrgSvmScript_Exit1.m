clc;
timeNow=clock;
disp(sprintf('startTime: %d/%d/%d %d:%d:%d',timeNow(1),timeNow(2),timeNow(3),timeNow(4),timeNow(5),round(timeNow(6))));
clear all;

addpath './SvmTools/';
addpath './ParameterPreProcessing/';
addpath './FileOperation/';
resultDirPrefix='./Result-Exit1/';


%% data input
train_data_name='../Data/data_ML_Exit1_Train.csv';
test_data_name='../Data/data_ML_Exit1_Test.csv';
trainData = csvread(train_data_name);
testData = csvread(test_data_name);
trainData(1,:)=[];
trainData(:,1)=[];
testData(1,:)=[];
testData(:,1)=[];
dataDimension=size(trainData,2);
trainLabel=trainData(:,dataDimension);
testLabel=testData(:,dataDimension);
trainData=trainData(:,1:dataDimension-1);
testData=testData(:,1:dataDimension-1);

inputData.trainGroup.data=trainData;
inputData.trainGroup.classLabel=trainLabel;
inputData.testGroup.data=testData;
inputData.testGroup.classLabel=testLabel;
clear trainData trainLabel testData testLabel


param.kernelType='RBF';
checkTest=true;
checkExp=true;


%% parameter prefix setting
if param.kernelType=='RBF'
    trainCmd='-s 0 -t 2 -h 0 -b 0 -q ';
elseif param.kernelType=='Linear'
    trainCmd='-s 0 -t 0 -h 0 -b 0 -q ';
end
testCmd='-b 0 -q';
param.gamma=-1;
param.cost=-1;
disp(sprintf(['TrainData:%d, TestData:%d'],size(inputData.trainGroup.data,1),size(inputData.testGroup.data,1)));
disp(sprintf(['OrgSVC with Kernel:',param.kernelType]));

%% search parameter && vectorization
gammaSeries=-10:2:10;
gammaSeries=exp(gammaSeries');

costSeries=10:-2:-10;
costSeries=exp(costSeries');

parameterSeries=ParameterWorkerVectorization(gammaSeries,costSeries,1);
fullParamBitLength=length(num2str(length(parameterSeries)));


param=repmat(param,[length(parameterSeries),1]);


%% main implementation part
% parameter travelling
% paramDir=[backupDataPrefix,nistDir];
DirMake(resultDirPrefix);
resultDir=[resultDirPrefix,'SvmOrg/'];
DirMake(resultDir);

trainGroup=inputData.trainGroup;
if checkTest
    testGroup=inputData.testGroup;
else
    testGroup=inputData.trainGroup;
end
disp(sprintf(['ParamNum:%d'],size(parameterSeries,2)));

    

parfor parameterTraveller=1:length(parameterSeries)
    tic;
    
    % generate param bup file name with path
    if checkExp
        thisGammaStr=['GammaExp',num2str(log(parameterSeries(parameterTraveller).gamma))];
        thisCostStr=['CostExp',num2str(log(parameterSeries(parameterTraveller).cost))];
    else
        thisGammaStr=['Gamma',num2str(parameterSeries(parameterTraveller).gamma)];
        thisCostStr=['Cost',num2str(parameterSeries(parameterTraveller).cost)];
    end

    thisParamStr(parameterTraveller)={['OrgSvm_',thisGammaStr,'_',thisCostStr,'.mat']};
    thisParamBupFileFullPath(parameterTraveller)={[resultDir,cell2mat(thisParamStr(parameterTraveller))]};
    
    
    % check file existence
    fileExists=MatFileFind(cell2mat(thisParamBupFileFullPath(parameterTraveller)));
    if fileExists
        paramBupOutput=ResultFileLoad(cell2mat(thisParamBupFileFullPath(parameterTraveller)));
        loadedAcry=paramBupOutput.acry;
        loadedAuc=paramBupOutput.auc;
        
        disp(sprintf([cell2mat(thisParamStr(parameterTraveller)),',ParaNo:%d/%d,Acry:%f,Auc:%f,timeElaps:%f'], ...,
        parameterTraveller,length(parameterSeries),loadedAcry,loadedAuc,toc));
        continue;
    end

    % find this parameter
    thisGamma=parameterSeries(parameterTraveller).gamma;
    thisCost=parameterSeries(parameterTraveller).cost;
    param(parameterTraveller).gamma=thisGamma;
    param(parameterTraveller).cost=thisCost;
    
    
    thisGammaStr=num2str(thisGamma);
    thisCostStr=num2str(thisCost);
    
    % generate train cmd
    thisGammaStr=['-g ',thisGammaStr,' '];
    thisCostStr=['-c ',thisCostStr,' '];
    thisTrainCmd=[trainCmd,thisGammaStr,thisCostStr];
    

    % overall svm
    [orgAcry, orgAuc, orgR2, classifier]=OrgSvmImplementation(trainGroup,testGroup,thisTrainCmd,testCmd,param(parameterTraveller));
    
    % accuracy display
    disp(sprintf([cell2mat(thisParamStr(parameterTraveller)),',ParaNo:%d/%d,Acry:%f,Auc:%f,timeElaps:%f'], ...,
        parameterTraveller,length(parameterSeries),orgAcry,orgAuc,orgR2,toc));
    
    
    
    % data bup write
    ResultFileWriteForOrg(orgAcry,orgAuc, orgR2, classifier, cell2mat(thisParamBupFileFullPath(parameterTraveller)));
end


% check for org result
for parameterTraveller=1:length(parameterSeries)
    paramBupOutput=ResultFileLoad(cell2mat(thisParamBupFileFullPath(parameterTraveller)));
    loadedAcry=paramBupOutput.acry;
    loadedAuc=paramBupOutput.auc;
    outputAcry(parameterTraveller)=paramBupOutput.acry;
    outputAuc(parameterTraveller)=paramBupOutput.auc;
    outputR2(parameterTraveller)=paramBupOutput.r2;
    clear paramBupOutput;
end


% reshape for final result
max(outputAcry)
outputAcry=reshape(outputAcry,[length(costSeries),length(gammaSeries)]);

max(outputAuc)
outputAuc=reshape(outputAuc,[length(costSeries),length(gammaSeries)]);

max(outputR2)
outputR2=reshape(outputR2,[length(costSeries),length(gammaSeries)]);


% reshape for final result
if checkExp
    gammaSeries=log(gammaSeries);
    costSeries=log(costSeries);
end



for gammaTraveller=1:length(gammaSeries)
gammaMaxAcry(gammaTraveller)=max(max(squeeze(outputAcry(:,gammaTraveller))));
end
gammaMaxAcry=gammaMaxAcry';
gammaMaxAcryValue=max(gammaMaxAcry);
gammaMaxAcryIndices=find(gammaMaxAcry==gammaMaxAcryValue);
gammaMaxAcry_GammaValue=gammaSeries(gammaMaxAcryIndices);

for costTraveller=1:length(costSeries)
costMaxAcry(costTraveller)=max(max(squeeze(outputAcry(costTraveller,:))));
end
costMaxAcry=costMaxAcry';
costMaxAcryValue=max(costMaxAcry);
costMaxAcryIndices=find(costMaxAcry==costMaxAcryValue);
costMaxAcry_CostValue=costSeries(costMaxAcryIndices);


for gammaTraveller=1:length(gammaSeries)
gammaMaxAuc(gammaTraveller)=max(max(squeeze(outputAuc(:,gammaTraveller))));
end
gammaMaxAuc=gammaMaxAuc';
gammaMaxAucValue=max(gammaMaxAuc);
gammaMaxAucIndices=find(gammaMaxAuc==gammaMaxAucValue);
gammaMaxAuc_GammaValue=gammaSeries(gammaMaxAucIndices);


for costTraveller=1:length(costSeries)
costMaxAuc(costTraveller)=max(max(squeeze(outputAuc(costTraveller,:))));
end
costMaxAuc=costMaxAuc';
costMaxAucValue=max(costMaxAuc);
costMaxAucIndices=find(costMaxAuc==costMaxAucValue);
costMaxAuc_CostValue=costSeries(costMaxAucIndices);


for gammaTraveller=1:length(gammaSeries)
gammaMaxR2(gammaTraveller)=max(max(squeeze(outputR2(:,gammaTraveller))));
end
gammaMaxR2=gammaMaxR2';
gammaMaxR2Value=max(gammaMaxR2);
gammaMaxR2Indices=find(gammaMaxR2==gammaMaxR2Value);
gammaMaxR2_GammaValue=gammaSeries(gammaMaxR2Indices);


for costTraveller=1:length(costSeries)
costMaxR2(costTraveller)=max(max(squeeze(outputR2(costTraveller,:))));
end
costMaxR2=costMaxR2';
costMaxR2Value=max(costMaxR2);
costMaxR2Indices=find(costMaxR2==costMaxR2Value);
costMaxR2_CostValue=costSeries(costMaxR2Indices);
    
    




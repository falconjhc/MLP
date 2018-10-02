function outputData = PicesWiseDividing(inputData,fullCorrectTestFieldLabelVec,eachPieceFieldNum)
outputData.trainGroup=inputData.trainGroup;

fullTestGroup=inputData.testGroup;

fullFieldLabelVec=unique(fullTestGroup.fieldLabel);

groupCounter=2;
testPieceMark=[];
transferTestFieldCounter=1;
noNeedTransferIndicesOnOrgOrder=[];


for fullFieldLabelVecTraveller=1:length(fullFieldLabelVec)
        
    thisFieldLabel=fullFieldLabelVec(fullFieldLabelVecTraveller);
    thisFieldRelatedIndices=find(fullTestGroup.fieldLabel==thisFieldLabel);
        
    if isempty(find(fullCorrectTestFieldLabelVec==thisFieldLabel))
        if (mod(transferTestFieldCounter,eachPieceFieldNum)==1) || (eachPieceFieldNum==1)
            outputData.testDataPieceList(groupCounter).data=[];
            outputData.testDataPieceList(groupCounter).classLabel=[];
            outputData.testDataPieceList(groupCounter).fieldLabel=[];
            outputData.testDataPieceList(groupCounter).indicesOnOrgOrder=[];
        end   
    
        thisFieldRelatedData=fullTestGroup.data(thisFieldRelatedIndices,:);
        thisFieldRelatedClassLabel=fullTestGroup.classLabel(thisFieldRelatedIndices);
        thisFieldRelatedFieldLabel=fullTestGroup.fieldLabel(thisFieldRelatedIndices);
    
        outputData.testDataPieceList(groupCounter).data=[outputData.testDataPieceList(groupCounter).data;thisFieldRelatedData];
        outputData.testDataPieceList(groupCounter).classLabel=[outputData.testDataPieceList(groupCounter).classLabel;thisFieldRelatedClassLabel];
        outputData.testDataPieceList(groupCounter).fieldLabel=[outputData.testDataPieceList(groupCounter).fieldLabel;thisFieldRelatedFieldLabel];
        outputData.testDataPieceList(groupCounter).indicesOnOrgOrder=[outputData.testDataPieceList(groupCounter).indicesOnOrgOrder;thisFieldRelatedIndices];
    
        testPieceMark=[testPieceMark;groupCounter*ones(length(thisFieldRelatedIndices),1)];
    
        if (mod(transferTestFieldCounter,eachPieceFieldNum)==0) || (eachPieceFieldNum==1)
            groupCounter=groupCounter+1;
        end
            
        transferTestFieldCounter=transferTestFieldCounter+1;
            
    else
       testPieceMark=[testPieceMark;-1*ones(length(thisFieldRelatedIndices),1)];
       noNeedTransferIndicesOnOrgOrder=[noNeedTransferIndicesOnOrgOrder;thisFieldRelatedIndices];
    end
end


correctEstmIndices=find(testPieceMark==-1);
outputData.testDataPieceList(1).data=fullTestGroup.data(correctEstmIndices,:);
outputData.testDataPieceList(1).classLabel=fullTestGroup.classLabel(correctEstmIndices);
outputData.testDataPieceList(1).fieldLabel=fullTestGroup.fieldLabel(correctEstmIndices);
outputData.testDataPieceList(1).indicesOnOrgOrder=noNeedTransferIndicesOnOrgOrder;


% sd3Length=length(inputData.sd3testGroup.fieldLabel);
% sd7Length=length(inputData.sd7testGroup.fieldLabel);
% 
% fullTestGroupMarks(1:sd3Length)=0;
% fullTestGroupMarks(sd3Length+1:sd3Length+sd7Length)=1;
% outputData.testGroupMarks=fullTestGroupMarks';
outputData.testDataPieceMarks=testPieceMark;

outputData.trueClassLabel_orgOrder=fullTestGroup.classLabel;
outputData.trueClassLabel_dataPieceOrder=[];
for dataPieceTraveller=1:length(outputData.testDataPieceList)
    outputData.trueClassLabel_dataPieceOrder=[outputData.trueClassLabel_dataPieceOrder;outputData.testDataPieceList(dataPieceTraveller).classLabel];
end







end


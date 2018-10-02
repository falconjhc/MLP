function [ nonTransferOutput ] = NonTransferInputEstmLabelPreProcess(nonTransferInput,fullTrueClassLabel,testPieceMarks)
tdrLabel=cell2mat(nonTransferInput.fld.tdr.estmLabel);
vdrLabel=cell2mat(nonTransferInput.fld.vdr.estmLabel);

% tdrProbabilities=ProbabilitiesNormalization(1,cell2mat(nonTransferInput.fld.tdr.votingResult));
% vdrProbabilities=ProbabilitiesNormalization(1,cell2mat(nonTransferInput.fld.vdr.votingResult));



testPieceMarkVec=unique(testPieceMarks);
for testPieceMarksTraveller=1:length(testPieceMarkVec)
    thisMark=testPieceMarkVec(testPieceMarksTraveller);
    thisRelatedIndices=find(testPieceMarks==thisMark);
    thisTdrLabel=tdrLabel(thisRelatedIndices);
    thisVdrLabel=vdrLabel(thisRelatedIndices);
    thisTrueLabel=fullTrueClassLabel(thisRelatedIndices);
    
%     thisTdrProbabilities=tdrProbabilities(thisRelatedIndices,:);
%     thisVdrProbabilities=vdrProbabilities(thisRelatedIndices,:);
    
    tdrLabelList(testPieceMarksTraveller)={thisTdrLabel};
    vdrLabelList(testPieceMarksTraveller)={thisVdrLabel};
    trueLabelList(testPieceMarksTraveller)={thisTrueLabel};
    
    tdrProbabilitiesList(testPieceMarksTraveller)={-1};
    vdrProbabilitiesList(testPieceMarksTraveller)={-1};
    
    tdrAcry(testPieceMarksTraveller)=length(find(thisTdrLabel==thisTrueLabel))/length(thisTrueLabel)*100;
    vdrAcry(testPieceMarksTraveller)=length(find(thisVdrLabel==thisTrueLabel))/length(thisTrueLabel)*100;
    
end

nonTransferOutput.tdrEstmLabelList=tdrLabelList;
nonTransferOutput.vdrEstmLabelList=vdrLabelList;
nonTransferOutput.trueLabelList=trueLabelList;

nonTransferOutput.tdrAcry=tdrAcry;
nonTransferOutput.vdrAcry=vdrAcry;

nonTransferOutput.tdrProbabilities=tdrProbabilitiesList;
nonTransferOutput.vdrProbabilities=vdrProbabilitiesList;


end


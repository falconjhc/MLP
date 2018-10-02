function [ nonTransferOutput ]=NonTransferInputProbabilitiesPreProcess(prevOutput,nonTransferInput,testPieceMarks,softmaxProbabilities)
nonTransferOutput=prevOutput;
tdrProbabilities=ProbabilitiesNormalization(softmaxProbabilities,cell2mat(nonTransferInput.fld.tdr.votingResult));
vdrProbabilities=ProbabilitiesNormalization(softmaxProbabilities,cell2mat(nonTransferInput.fld.vdr.votingResult));



testPieceMarkVec=unique(testPieceMarks);
for testPieceMarksTraveller=1:length(testPieceMarkVec)
    thisMark=testPieceMarkVec(testPieceMarksTraveller);
    thisRelatedIndices=find(testPieceMarks==thisMark);
    
    
    thisTdrProbabilities=tdrProbabilities(thisRelatedIndices,:);
    thisVdrProbabilities=vdrProbabilities(thisRelatedIndices,:);
    
    
    tdrProbabilitiesList(testPieceMarksTraveller)={thisTdrProbabilities};
    vdrProbabilitiesList(testPieceMarksTraveller)={thisTdrProbabilities};
    
    
end

nonTransferOutput.tdrProbabilities=tdrProbabilitiesList;
nonTransferOutput.vdrProbabilities=vdrProbabilitiesList;
end


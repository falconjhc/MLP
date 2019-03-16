function ResultFileWriteForOrg(acry,auc,r2, classifier,fullSavePath)
paramBupOutput.acry=acry;
paramBupOutput.auc=auc;
paramBupOutput.r2=r2;
paramBupOutput.classifier=classifier;


variableName=who('paramBupOutput');
variableName=cell2mat(variableName(1));
save(fullSavePath,variableName);
end
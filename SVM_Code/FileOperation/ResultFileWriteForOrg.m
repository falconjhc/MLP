function ResultFileWriteForOrg(acry,auc,classifier,fullSavePath)
paramBupOutput.acry=acry;
paramBupOutput.auc=auc;
paramBupOutput.classifier=classifier;

variableName=who('paramBupOutput');
variableName=cell2mat(variableName(1));
save(fullSavePath,variableName);
end
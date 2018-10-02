function ResultFileWriteForOrg(acry,auc,fullSavePath)
paramBupOutput.acry=acry;
paramBupOutput.auc=auc;

variableName=who('paramBupOutput');
variableName=cell2mat(variableName(1));
save(fullSavePath,variableName);
end
function ParamBupFileWriteForFld_FieldSpecificTransfer(fldPfmc,fldObj,nonTransferBup,fullSavePath)
paramBupOutput.fldPfmc_fst=fldPfmc;
paramBupOutput.fldObj_fst=fldObj;
paramBupOutput.fldPfmc_nt=nonTransferBup.fldPfmc;
paramBupOutput.fldObj_nt=nonTransferBup.fldObj;
paramBupOutput.org=nonTransferBup.org;


variableName=who('paramBupOutput');
variableName=cell2mat(variableName(1));
save(fullSavePath,variableName);
end
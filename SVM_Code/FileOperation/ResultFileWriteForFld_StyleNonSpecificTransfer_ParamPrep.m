function ResultFileWriteForFld_StyleNonSpecificTransfer_ParamPrep(classPairs,org,tdr,vdr,fullSavePath)
paramBupOutput.org=org;
paramBupOutput.fld.tdr=tdr;
paramBupOutput.fld.vdr=vdr;

variableName=who('paramBupOutput');
variableName=cell2mat(variableName(1));
save(fullSavePath,variableName);
end
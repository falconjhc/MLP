function ResultFileWriteForFld_StyleNonSpecificTransfer(classPairs,org,tdr,vdr,obj,performanceCheck,fullSavePath)
paramBupOutput.org=org;
paramBupOutput.fld.tdr=tdr;
paramBupOutput.fld.vdr=vdr;
paramBupOutput.fld.obj=obj;
paramBupOutput.performanceCheck=performanceCheck;


variableName=who('paramBupOutput');
variableName=cell2mat(variableName(1));
save(fullSavePath,variableName);
end
function ParamBupFileWriteForFld_NonTransfer(classPairs,org,tdr,vdr,obj,fullSavePath)
paramBupOutput.org=org;
paramBupOutput.fld.tdr=tdr;
paramBupOutput.fld.vdr=vdr;
paramBupOutput.fld.obj=obj;


variableName=who('paramBupOutput');
variableName=cell2mat(variableName(1));
save(fullSavePath,variableName);
end
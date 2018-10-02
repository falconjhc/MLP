function ResultFileWriteForFld_NonTransfer(org,tdr,vdr,fdr,obj,maxExpansionRate,bestClassifiers,fullSavePath)
paramBupOutput.org=org;
paramBupOutput.fld.tdr=tdr;
paramBupOutput.fld.vdr=vdr;
paramBupOutput.fld.fdr=fdr;
paramBupOutput.fld.obj=obj;
paramBupOutput.fld.expansionRate=maxExpansionRate;
paramBupOutput.fld.bestClassifiers=bestClassifiers;



variableName=who('paramBupOutput');
variableName=cell2mat(variableName(1));
save(fullSavePath,variableName);
end
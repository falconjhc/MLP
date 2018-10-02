function DataPieceResultFileSave( tdr,vdr,obj,path )


result.tdr=tdr;
result.vdr=vdr;
result.obj=obj;

variableName=who('result');
variableName=cell2mat(variableName(1));
save(path,variableName);


end


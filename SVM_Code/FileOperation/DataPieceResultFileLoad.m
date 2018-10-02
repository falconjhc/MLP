function [tdr,vdr,obj]=DataPieceResultFileLoad(path)

result=load(path);
result=result.result;



tdr=result.tdr;
vdr=result.vdr;
obj=result.obj;




end


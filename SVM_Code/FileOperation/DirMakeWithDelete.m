function DirMakeWithDelete(path)
if exist(path,'dir')~=0
    rmdir(path,'s');
end
mkdir(path);
end
function fullCorrectFieldLabelVec = FindFullCorrectTestFields(inputData,nonTransferResult)
fieldLabel=inputData.testGroup.fieldLabel;
trueClassLabel=inputData.testGroup.classLabel;

tdrAcry=nonTransferResult.fld.tdr.acry;
vdrAcry=nonTransferResult.fld.vdr.acry;
tdrEstmClassLabel=cell2mat(nonTransferResult.fld.tdr.estmLabel);
vdrEstmClassLabel=cell2mat(nonTransferResult.fld.vdr.estmLabel);

if tdrAcry>vdrAcry
    estmClassLabel=tdrEstmClassLabel;
else
    estmClassLabel=vdrEstmClassLabel;
end

fullCorrectFieldLabelVec=[];
fieldLabelVec=unique(fieldLabel);
for fieldLabelTraveller=1:length(fieldLabelVec)
    thisFieldLabel=fieldLabelVec(fieldLabelTraveller);
    thisFieldIndices=find(fieldLabel==thisFieldLabel);
    thisFieldTrueClassLabel=trueClassLabel(thisFieldIndices);
    thisFieldEstmClassLabel=estmClassLabel(thisFieldIndices);
    
    correctEstmClassLabelIndices=find(thisFieldTrueClassLabel==thisFieldEstmClassLabel);
    
    if length(correctEstmClassLabelIndices)==length(thisFieldTrueClassLabel)
        fullCorrectFieldLabelVec=[fullCorrectFieldLabelVec;thisFieldLabel];
    end
    
end


end


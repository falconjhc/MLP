function parameterSeries=ParameterWorkerVectorizationForTradeoffOnly(orgParameterSeries,tradeOffSeries)
counter=1;
for orgParamTraveller=1:length(orgParameterSeries)
    for tradeOffTraveller=1:length(tradeOffSeries)
        parameterSeries(counter).gamma=orgParameterSeries(orgParamTraveller).gamma;
        parameterSeries(counter).cost=orgParameterSeries(orgParamTraveller).cost;
        parameterSeries(counter).softmaxParam=orgParameterSeries(orgParamTraveller).softmaxParam;
        parameterSeries(counter).tradeOff=tradeOffSeries(tradeOffTraveller);
        counter=counter+1;
    end
end
tmpParameterMatrix=reshape(parameterSeries,[length(tradeOffSeries),length(orgParameterSeries)]);
end
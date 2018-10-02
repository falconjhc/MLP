function parameterSeries=ParameterWorkerVectorization(gammaSeries,costSeries,softmaxParamSeries)
counter=1;
for gammaTraveller=1:length(gammaSeries)
    for costTraveller=1:length(costSeries)
        for softMaxParamSeries=1:length(softmaxParamSeries)
            parameterSeries(counter).gamma=gammaSeries(gammaTraveller);
            parameterSeries(counter).cost=costSeries(costTraveller);
            parameterSeries(counter).softmaxParam=softmaxParamSeries(softMaxParamSeries);
            counter=counter+1;
        end
            
        
    end
end
tmpParameterMatrix=reshape(parameterSeries,[length(softmaxParamSeries),length(costSeries),length(gammaSeries)]);
end
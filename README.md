# CI_and_Prediction_Bootstraps
This module contains functions, bootStrapParamCI and bootStrapPredictInterval, that follow a bootstrap approach to produce confidence intervals for model parameters and prediction intervals for individual point predictions, respectively.

    bootStrapParamCI will run bootstrap a linear regression model nBootStraps times with n = sampleSize, and confidence level = confLevel
    If displayPlot==True and a model parameter is specified, then the coefficients from that model will be plotted on the CI plot that is printed
    if displayPlot==True, then a matplotlib plot of the confidence interval for each estimated coefficient will be displayed when bootStrapPramCI is called
    A dictionary of dictionaries is returned, with each entry being of form parameter:{'meanCoef', 'CI', 'significant'}.
    meanCoef is the mean estimate of the coefficient for that parameter, CI is the confidence interval for
    that parameter, estimated for n=sampleSize and confLevel = .9.
    'significant' signifies whether 0 is contained by CI. If 0 is not in CI, then 'significant'==True.
    The idea of the 'significant' key is that it can be used to determine if a parameter gets 
    included in a final model after running these bootstraps.
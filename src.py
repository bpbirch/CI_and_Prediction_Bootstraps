#%%
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
import scipy 
import sklearn
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import pickle
#%%
def bootStrapParamCI(df, features, target, sampleSize=None, confLevel=.9, nBootStraps=100, displayPlot=False, model=None):
    """
    bootStrapParamCI will run bootstrap a linear regression model nBootStraps times with n = sampleSize, and confidence level = confLevel
    If displayPlot==True and a model parameter is specified, then the coefficients from that model will be plotted on the CI plot that is printed
    if displayPlot==True, then a matplotlib plot of the confidence interval for each estimated coefficient will be displayed when bootStrapPramCI is called
    A dictionary of dictionaries is returned, with each entry being of form parameter:{'meanCoef', 'CI', 'significant'}.
    meanCoef is the mean estimate of the coefficient for that parameter, CI is the confidence interval for
    that parameter, estimated for n=sampleSize and confLevel = .9.
    'significant' signifies whether 0 is contained by CI. If 0 is not in CI, then 'significant'==True.
    The idea of the 'significant' key is that it can be used to determine if a parameter gets 
    included in a final model after running these bootstraps.

    Args:
        df (pd.DataFrame): pd DataFrame 
        features (list): list of dataframe featuresm, in form of strings
        target (str): target variable to regress on features
        sampleSize (int, optional): sample size to draw for each bootstrap. Defaults to None.
        confLevel (float, optional): confidence level. Defaults to .9.
        nBootStraps (int, optional): number of bootstraps to be run. feature coefficients are estimated for each bootstrap. Defaults to 100.
        displayPlot (bool, optional): if True, then a matplotlib histogram will be displayed for each coefficient CI when bootStrapParamCI is called. Defaults to False.
        model (sklearn.linear_model.LinearRegression(), optional): model whose parameters you can compare to bootstrapped models. Defaults to None.

    Raises:
        ValueError: confidence level cannot be larger than 1

    Returns:
        ciDict (dict): A dictionary of dictionaries is returned, with each entry being of form parameter:{'meanCoef', 'CI', 'significant'}.
                        meanCoef is the mean estimate of the coefficient for that parameter, CI is the confidence interval for
                        that parameter, estimated for n=sampleSize and confLevel = .9.
                        'significant' signifies whether 0 is contained by CI. If 0 is not in CI, then 'significant'==True.
                        The idea of the 'significant' key is that it can be used to determine if a parameter gets 
                        included in a final model after running these bootstraps.
    """
    if confLevel > 100:
        raise ValueError('confidence level cannot be higher than 100')
    fullData = [target] + features
    fullDf = df[fullData].copy()
    # now draw random records from our df sampleSize time:
    coeffDict = {}
    ciDict = {}
    lowQuantile = (1-confLevel)/2
    highQuantile = confLevel + (1-confLevel)/2
    if displayPlot:
        fig, ax = plt.subplots(1,len(features)) # if display==True, we will display plots of confidence intervals for parameters
        axIndex = 0
    if sampleSize == None:
        sampleSize = len(df.index)
    for feature in features:
        coeffDict[feature] = [] #initialize an empty list for each feature parameter
    for _ in range(nBootStraps):
        sampleDf = pd.DataFrame(columns=fullData)
        for i in range(sampleSize):
            sampleDf.loc[i] = fullDf.iloc[np.random.randint(0, len(fullDf.index)-1)] # .index will be 1 longer than .iloc
        # now create a model for that sampleDf and record the parameter vals
        model = LinearRegression()
        model.fit(sampleDf[features], sampleDf[target])
        for feature, coef in zip(features, model.coef_):
            coeffDict[feature].append(coef) # now append coefficients to coeffDict
    for listOfCoefs in coeffDict: 
        coefDf = pd.DataFrame(coeffDict[listOfCoefs], columns=[listOfCoefs])
        low = coefDf[listOfCoefs].quantile(lowQuantile)
        high = coefDf[listOfCoefs].quantile(highQuantile)
        if displayPlot:
            ax[axIndex] = coefDf.plot.hist(color='gray')
            ax[axIndex].set_xlabel(f'{listOfCoefs} parameter estimates')
            ax[axIndex].axvline(low, color='red', label=f'{confLevel} conf level bounds')
            ax[axIndex].axvline(high, color='red')
            if model:
                ax[axIndex].axvline(model.coef_[axIndex], label=f'input model coef estimate for {listOfCoefs}', color='green')
            else:
                ax[axIndex].axvline(np.mean(coeffDict[listOfCoefs]), label=f'mean coef estimate for {listOfCoefs}', color='green')
            ax[axIndex].legend()
            axIndex += 1
        ciDict[listOfCoefs] = {'meanCoef':np.mean(coeffDict[listOfCoefs]),
                                'CI': (low, high),
                                'significant': not low <= 0 <= high} # if 0 is in CI, then param is not significant
    # if displayPlot:
    #     plt.show()
    return ciDict

if __name__ == '__main__':
    # working with some housing data, regressing adjSalePrice on the following features
    df = pd.read_csv('https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/house_sales.csv',
                delim_whitespace=True)
    features = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
    target = 'AdjSalePrice'
    fullData = [target] + features

    df1 = df[fullData].copy()

    # if we wanted to use weighted data:
    # df1['Year'] = [int(date.split('-')[0]) for date in df.DocumentDate] 
    # df1['Weight'] = df1.Year - 2005 # let's weight based on how old our data is. Older data is less relevant

    # let's fit one individual model
    houseModel = LinearRegression()
    houseModel.fit(df1[features], df1[target]) # weight our model based on age of data
    confIntDict = bootStrapParamCI(df1, features, target, sampleSize=100, displayPlot=True, model=houseModel)
    print(confIntDict)
    # we can see that with sample size of n=100, only BldgGrade and SqFtTotLiving are statistically significant,
    # (meaning 0 falls within the CI of the other coefficients)
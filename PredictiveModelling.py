''' Predictive Modelling Task
    
    To run: type "python PredictiveModelling.py" in your command prompt and hit enter
    
    THis script will produce:
    1. 3-day forecasted return for a timestamp in the future using the last-day closing price 
       provided for each stock in the dataset
    2. Effect of using additional signals and combination of additional signals on the long-term
       metrics of fitted-model for each stock in the dataset
     
    
'''

import pandas as pd
import numpy as np
import warnings
# Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller
# auto corelation
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
# seasonal decompose
from statsmodels.tsa.seasonal import seasonal_decompose
# ARIMA
import statsmodels.tsa as tsa
# display
from IPython.display import display
import colorama
import matplotlib.pyplot as plt



def testStationarity(ts=None, displaySummary=True, displayPlot=True):
    ''' Test whether the input series is stationary or not
    '''
    # remove NAN's
    ts = ts.dropna()
    
    # create a data frame for X1
    dframe = pd.DataFrame(ts.index)
    dframe['P(t)'] = ts.values
    d = dframe['P(t)'][~dframe['P(t)'].isnull()]

    # dickyey-fuller test
    df = adfuller(ts)
    if df[0] < df[4]['1%']:
        confi = 0.99
        isStationary = True;
        strStationary = ' DFTest: Stationary'+ ' (confidence= %.2f)' % confi
        
    elif df[0] < df[4]['5%']:
        confi = 0.95
        isStationary = True;
        strStationary = ' DFTest: Stationary'+ ' (confidence= %.2f)' % confi
        
    elif df[0] < df[4]['10%']:
        confi = 0.90
        isStationary = True;
        strStationary = ' DFTest: Stationary' + ' (confidence= %.2f)' % confi
        
    else:
        confi = 0
        isStationary = False;
        strStationary = ' DFTest: Non Stationary'
        
    # moving averages
    dframe['MA_30'] = pd.Series.rolling(d, center=False, window=30).mean() # 30 -day MA
    dframe['MA_60'] = pd.Series.rolling(d, center=False, window=60).mean() # 60 -day MA
    dframe['MA_120'] = pd.Series.rolling(d, center=False, window=120).mean() # 120 -day MA

    # standard devs
    dframe['SD_30'] = pd.Series.rolling(d, center=False, window=30).std() # 30 -day SD
    dframe['SD_60'] = pd.Series.rolling(d, center=False, window=60).std() # 60 -day SD
    dframe['SD_120'] = pd.Series.rolling(d, center=False, window=120).std() # 120 -day SD
    
    if displayPlot:
        # plot
        dframe.loc[:, ['MA_30', 'MA_60', 'MA_120']].plot( title='Moving Averages - ' + ts.name + strStationary)
        dframe.loc[:, ['SD_30', 'SD_60', 'SD_120']].plot( title='Variance - ' + ts.name + strStationary)
        
    if displaySummary:
        print(df)
        print('Time series - '+ts.name + ',' + strStationary)
        
    return isStationary


def convert2Stationary(ts=None, disp=True):
    ''' Convert a non-stationary time-series to stationary using simple differencing of
        log-transformed timeseries
        The input time-series is first checked for stationarity then the conversion is done
        
        input: ts -> time-series in normal price
        output: stationary timeseries
    '''
    # log
    ts_log = np.log(ts.dropna())
    out = ts_log
    if not testStationarity(ts, displayPlot=False, displaySummary=False):
        # differencing
        out = ts_log - ts_log.shift(1)
    
    # perform test
    if disp:
        print("=== after conversion test for Stationarity===")
        testStationarity(out, displayPlot=disp, displaySummary=disp)
        
    return out



def forecast_3day_returns(ts=None, ARIMAorder=None, exogVariable=None, displaySummary=True):
    ''' Forecast 3-day returns using ARIMA modelling
    '''
    # original timeseries
    orig_tseries = ts
    # create a new time series by pushing it 3-days ahead and fill it with 0
    pushed_tseries = pd.Series(0, orig_tseries.index + pd.Timedelta('3 day'))
    # extract the last 3-days from pushed_tseries and append it to original timeseries
    tseries = orig_tseries.append(pushed_tseries[-3 : pushed_tseries.shape[0]], verify_integrity=True)
    
    # convert original timeseries X3 to stationary
    day_returns = convert2Stationary(ts, False)
    # index-0 of day_returns will be NAN due to differencing.
    # usually day_returns[original_length+1] will have -INF due to 0 insertion.

    # ARIMA order
    if ARIMAorder is None:
        ARIMAorder = (1,0,1)
    
    # Exogenous variable
    exogArrValues = None
    doexit = False
    if exogVariable is not None :
        # make sure sizes are same as original series
        assert(orig_tseries.shape[0] == exogVariable.shape[0])
        ## need to drop rows from exogVariable that correspond to rows in original-series with NAN in it
        # to do this, first create a dummy with NAN if NAN in original , 1 otherwise, 
        # then, multiple dummy to exogVariable, thereby replacing NAN if NAN in original, value otherwise
        # finally, dropna
        dummy = (ts * (ts.isnull() * 1)) + 1
        exogVariable = exogVariable.multiply(dummy, axis='rows')
        exogVariable = exogVariable.dropna()
        # make sure after dropping NAN sizes are same as day_returns
        assert(day_returns.shape[0] == exogVariable.shape[0])
        if 0 in exogVariable:
            # check if dummy-variable added by fillExogenousVariable has all 1's
            # dummy column name is 0, if true, then it causes an exception in ARIMA modelling, strip it off
            if sum(exogVariable[0]) == exogVariable.shape[0]:
                exogVariable = exogVariable.drop(0, axis=1)

            # check if dummy-variable dded by fillExogenousVariable has all 0's, then set exogVariable = None 
            # or issue an ERROR
            # dummy column name is 0, if true, then it causes an exception in ARIMA modelling, strip it off
            elif sum(exogVariable[0]) == 0:
                #exogVariable = exogVariable.drop(0, axis=1)
                print(colorama.Fore.RED + "WARNING: Exogeneous variable does not any values associated with any timestamps of the Input series")
                doexit = True

            # day_returns[1:] is the input for modelling, so make sure exogVariable gets the same treatment
        
        exogArrValues = exogVariable[1:]

    if doexit:
        return
        
    # modelling (use all data starting from index-1 day of day_returns to the lastday of the orignal timeseries)
    model = tsa.arima_model.ARIMA(day_returns.ix[1:], order=ARIMAorder, exog=exogArrValues).fit(disp=-1)    
    if displaySummary:
        print(model.summary())

    # index of day_returns runs from 0:day_returns.shape[0]-1, there are total of day_returns.shape[0] elements
    # we are ignoring inedx-0 while training (it might contain NAN due to differencing), hence we are only training
    # with day_returns.shape[0]-1 elements (with indexing running from 1:day_returns.shape[0]-1). 
    # Therefore the the trained-model will have a length of ((day_returns.shape[0]-1)-1)
    # For prediction, the starting index should atleast be the last fittedvalue.

    # predict
    preds = model.predict(model.fittedvalues.shape[0]-1, model.fittedvalues.shape[0]+2, exogVariable)

    ### calculate error metrics and accuracy of fittedvalues
    # Original-returns: calculate rolling_sum of log-returns over a window of 4 days, including current day
    orig_returns = day_returns[1:]
    rollsum = pd.Series.rolling(orig_returns, center=False, window=4).sum()
    orig_3day_returns =  orig_returns - rollsum.shift(-3)
    # fitted-returns: 
    rollsum = pd.Series.rolling(model.fittedvalues, center=False, window=4).sum()
    fitted_3day_returns =  model.fittedvalues - rollsum.shift(-3)
    # display dataframe

    statsFrame = pd.DataFrame(orig_returns.index)
    statsFrame['Orig Log Returns'] = orig_returns.values
    statsFrame['Pred Log Returns'] = model.fittedvalues.values
    statsFrame['AbsError'] = np.abs(orig_returns.values - model.fittedvalues.values)
    statsFrame['AbsPercentError(%)'] = np.abs((orig_returns.values - model.fittedvalues.values)/(orig_returns.values+0.00000001)) * 100 
    statsFrame['DirectionAccuracy'] = ~((orig_returns.values > 0) != (model.fittedvalues.values > 0))
    statsFrame['Orig 3-day Log returns'] = orig_3day_returns.values
    statsFrame['Pred 3-day Log returns'] = fitted_3day_returns.values
    statsFrame['3-day AbsError'] = np.abs(orig_3day_returns.values - fitted_3day_returns.values)
    statsFrame['3-day AbsPercentError(%)'] = np.abs((orig_3day_returns.values - fitted_3day_returns.values)/(orig_3day_returns.values+0.00000001)) * 100 
    statsFrame['3-day DirectionAccuracy'] = ~((orig_3day_returns.values > 0) != (fitted_3day_returns.values > 0))
    # calculate


    # create a fitted_day_returns series using index from new timeseries (created used pushed_timeseries) and fittedvalues
    fitted_day_returns = pd.Series(model.fittedvalues, index=tseries.index)
    # copy the forecast values to correct dates
    fitted_day_returns[-4 : fitted_day_returns.shape[0]] = preds
    forecast_day_returns = fitted_day_returns[-4 : fitted_day_returns.shape[0]]

    # Predicted-returns: 
    rollsum = pd.Series.rolling(forecast_day_returns, center=False, window=4).sum()
    forecast_3day_returns =  forecast_day_returns - rollsum.shift(-3)

    # Display
    futureFrame = pd.DataFrame(forecast_day_returns.index)
    futureFrame['Forecast Log Returns'] = forecast_day_returns.values
    futureFrame['Forecast 3-day Log Returns'] = forecast_3day_returns.values
    futureFrame['Long term MAE (log returns)'] = np.mean(statsFrame['AbsError'])
    futureFrame['Long term MAPE (log returns) %'] = np.mean(statsFrame['AbsPercentError(%)']/100)*100
    futureFrame['Long term DirectionAccuracy (log returns) %'] = (sum(statsFrame['DirectionAccuracy'])/statsFrame.shape[0]) * 100
    futureFrame['Long term MAE (3-day log returns)'] = np.mean(statsFrame['3-day AbsError'])
    futureFrame['Long term MAPE (3-day log returns) %'] = np.mean(statsFrame['3-day AbsPercentError(%)']/100) * 100 
    futureFrame['Long term DirectionAccuracy (3-day log returns) %'] = (sum(statsFrame['3-day DirectionAccuracy'])/statsFrame.shape[0]) * 100
    if displaySummary:
        print("=====  3-day Returns forecast for closing price of last-day in timeseries ======")
        print("           Note: The first row in the table corresponds to the last day \n           closing price of the stock")
        print("   Stock: "+ ts.name)
        display(futureFrame)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    
    return futureFrame



def fillExogenousVariable(exvar=None):
    ''' Check if the exogenous variable series has NANs, concatenate dummy variable if required 
        if yes, then concat a dummy variable to help handle these NANs
    '''
    # dummy variable
    dummy = pd.Series(True, index=exvar.index)
    for col in exvar:
        dummy = np.logical_and(dummy, ~exvar[col].isnull())
        
    
    # replace NANs in exvar with 0
    ex = exvar
    ex[~dummy] = 0
    # convert bool to int
    dummy = dummy * 1
    # output
    out = pd.concat([ex, dummy], axis=1)
    return out



def compareForecasts(forecast_with=None, forecast_without=None):
    ''' Compare two forecast metrics and output some statistics
        Note: forecast_with and forecast_without should be outputs of the forecast_3day_returns() function
    '''
    if forecast_with is None:
        return
    
    if forecast_without is None:
        return
    
    fwith = forecast_with.iloc[0]
    fwithout = forecast_without.iloc[0]
    sta = ((fwith.values[1:] - fwithout.values[1:])/fwithout.values[1:]) * 100

    out = pd.DataFrame(forecast_with.index)
    out['Long Term MAE (log returns) % change'] = sta[2]
    out['Long Term MAPE (log returns) % change'] = sta[3]
    out['Long Term DirectionAccuracy (log returns) % change'] = sta[4]
    out['Long Term MAE (3-day log returns) % change'] = sta[5]
    out['Long Term MAPE (3-day log returns) % change'] = sta[6]
    out['Long Term DirectionAccuracy (3-day log returns) % change'] = sta[7]
    return out[:1]


if __name__ == "__main__":
   # convert data to time-series data while reading from csv itself
   dailyPrices = pd.read_csv('dailyPrices_AtClose.csv', parse_dates='t()', index_col='t()')
   #dailyPrices.index # check if the index is datetime
   # re-index and set frequency='D'
   dates = pd.date_range(dailyPrices.index[0], dailyPrices.index[-1], freq='D')
   dailyPrices = dailyPrices.reindex(dates)
   
   # convert data to time-series data while reading from csv itself
   x1AtClose = pd.read_csv('X1Signals_AtClose.csv', parse_dates='t()', index_col='t()')
   #x1AtClose.index # check if the index is datetime
   # re-index and set frequency='D'
   dates = pd.date_range(dailyPrices.index[0], dailyPrices.index[-1], freq='D')
   x1AtClose = x1AtClose.reindex(dates)

   # convert data to time-series data while reading from csv itself
   x2AtClose = pd.read_csv('X2Signals_AtClose.csv', parse_dates='t()', index_col='t()')
   #x1AtClose.index # check if the index is datetime
   # re-index and set frequency='D'
   dates = pd.date_range(dailyPrices.index[0], dailyPrices.index[-1], freq='D')
   x2AtClose = x2AtClose.reindex(dates)
   
   # process all stocks and generate 3-day log returns
   print('**********************************************************************************************')
   print('************                   3-DAY LOG RETURNS ON ALL STOCK                 ****************')
   print('**********************************************************************************************')
   for stock in dailyPrices:
	   print('*******************************    Stock :  ' + dailyPrices[stock].name + '   ****************************')
	   forecast_3day_returns(dailyPrices[stock], ARIMAorder=(1,0,0))

   print('**********************************************************************************************')
   print('**********************************************************************************************')
   print('**********************************************************************************************')
   
   # process all stocks to determine if additional signals have any effect
   print('**********************************************************************************************')
   print('************          EFFECT OF ADDITIONAL SIGNAL EXPERIMENTATION             ****************')
   print('**********************************************************************************************')
   for stock in dailyPrices:
	   print('*******************************    Stock :  ' + dailyPrices[stock].name + '   ****************************')
	   # forecast without additional signal
	   forecast_without = forecast_3day_returns(dailyPrices[stock], ARIMAorder=(1,0,0), displaySummary=False)
	   
	   # forecast with x1AtClose
	   # convert exogenous series to dataframe
	   exvar =[]
	   exvar = pd.DataFrame(x1AtClose[stock], index=x1AtClose.index)
	   exvar = fillExogenousVariable(exvar)
	   forecast_with1 = forecast_3day_returns(dailyPrices[stock], ARIMAorder=(1,0,0), exogVariable=exvar, displaySummary=False)
	   
	   # forecast with x2AtClose
	   # convert exogenous series to dataframe
	   exvar =[]
	   exvar = pd.DataFrame(x2AtClose[stock], index=x2AtClose.index)
	   exvar = fillExogenousVariable(exvar)
	   forecast_with2 = forecast_3day_returns(dailyPrices[stock], ARIMAorder=(1,0,0), exogVariable=exvar, displaySummary=False)
	   
	   ## forecast with x1AtClose & x2AtClose
	   # convert exogenous series to dataframe
	   exvar =[]
	   exvar = pd.DataFrame(x1AtClose[stock], index=x1AtClose.index)
	   exvar['X31'] = x2AtClose.X3
	   exvar = fillExogenousVariable(exvar)
	   forecast_with12 = forecast_3day_returns(dailyPrices[stock], ARIMAorder=(1,0,0), exogVariable=exvar, displaySummary=False)
	   
	   print('=============   X1Signal_AtClose as additional: Change in metrics with & without')
	   out = []
	   out = compareForecasts(forecast_with1, forecast_without)
	   display(out)
	   print('=============   X2Signal_AtClose as additional: Change in metrics with & without')
	   out =[]
	   out = compareForecasts(forecast_with2, forecast_without)
	   display(out)
	   print('=============   X1Signal_AtClose + X2Signal_AtClose as additional: Change in metrics with & without')
	   out =[]
	   out = compareForecasts(forecast_with12, forecast_without)
	   display(out)
	   print('*******************************    Stock :  ' + dailyPrices[stock].name + '   ****************************')
   
   print('**********************************************************************************************')
   print('**********************************************************************************************')
   print('**********************************************************************************************')
   

   

# MLTK Custom Algorithm - Forecasting


## Installation

Clone this repo under `$SPLUNK_HOME/etc/apps` alongside `$SPLUNK_HOME/etc/apps/Splunk_ML_Toolkit`.


## SystemIdentification algorithm

Syntax:

```
| fit SystemIdentification <target_field> from <feature_field> [lags=<relative_time>] [model=[LinearRegression|RandomForestRegressor]]
```

Example:

```
| inputlookup app_usage.csv
| fit SystemIdentification CRM HR1 from ERP Expenses HR2 ITOps CloudDrive RemoteAccess OTHER Recruiting lags="14d,15d,16d" time_field=_time model=LinearRegression into forecast_model
| table _time *CRM* *HR1*
```

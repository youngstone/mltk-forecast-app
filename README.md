# MLTK Custom Algorithm - Forecasting


## Installation

Clone this repo under `$SPLUNK_HOME/etc/apps` alongside `$SPLUNK_HOME/etc/apps/Splunk_ML_Toolkit`.


## SystemIdentification algorithm

Syntax:

```
| fit SystemIdentification <target_field> from <feature_field> [lags=<relative_time>] [model=[LinearRegression|RandomForestRegressor]]
```

`lags` will take form of `<time_integer><time_unit>`. The supported time units are listed below.

```
second: {'s', 'sec', 'secs', 'second', 'seconds'}
minute: {'m', 'min', 'minute', 'minutes'}
hour: {'h', 'hr', 'hrs', 'hour', 'hours'}
day: {'d', 'day', 'days'}
month: {'mon', 'month', 'months'}
```

Example query:

```
| inputlookup app_usage.csv
| fit SystemIdentification CRM HR1 from ERP Expenses HR2 ITOps CloudDrive RemoteAccess OTHER Recruiting lags="14d,15d,16d" time_field=_time model=LinearRegression
| table _time *CRM* *HR1*
```

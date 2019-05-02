#!/usr/bin/env python
import functools
import re
import copy
from datetime import timedelta

import pandas as pd
import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from codec import codecs_manager
from base import BaseAlgo, ClassifierMixin
from util.param_util import convert_params
from util import df_util


SECOND = {'s', 'sec', 'secs', 'second', 'seconds'}
MINUTE = {'m', 'min', 'minute', 'minutes'}
HOUR = {'h', 'hr', 'hrs', 'hour', 'hours'}
DAY = {'d', 'day', 'days'}
WEEK = {'w', 'week', 'weeks'}
MONTH = {'mon', 'month', 'months'}
QUARTER = {'q', 'qtr', 'qtrs', 'quarter', 'quarters'}
YEAR = {'y', 'yr', 'yrs', 'year', 'years'}

LAG_PATTERN = re.compile(r'(\d+)([a-z]+)')
TIME_FIELD = '_time'


def convert_lag(lag):
    time_delta = timedelta(0)
    for num, unit in LAG_PATTERN.findall(lag.lower()):
        num = int(num)
        if unit in MONTH:
            time_delta += timedelta(month=num)
        elif unit in DAY:
            time_delta += timedelta(days=num)
        elif unit in HOUR:
            time_delta += timedelta(hours=num)
        elif unit in MINUTE:
            time_delta += timedelta(minutes=num)
        elif unit in SECOND:
            time_delta += timedelta(seconds=num)
        else:
            msg = "Unrecognized lag format: \"{}\"".format(unit)
            raise Exception()
    return time_delta


class LagHandler(object):
    
    def __init__(self, lag):
        self.lag = lag
        self.df_lagged = pd.DataFrame()

    def process_df(self, df, feature_fields, target_fields, time_field):
        relevant_fields = feature_fields + target_fields
        all_fields = relevant_fields + [time_field]
        df_lagged = df[all_fields].copy(deep=True)
        time_delta = convert_lag(self.lag)
        df_lagged[time_field] = df_lagged[time_field] + time_delta
        lagged_fields = [col+'@'+self.lag for col in relevant_fields]
        df_lagged.rename(columns={col: new_col for col, new_col in zip(relevant_fields, lagged_fields)}, inplace=True)
        return df_lagged, lagged_fields


def handle_LinearRegression(options):

    out_params = convert_params(
        options.get('params', {}), 
        bools=['fit_intercept', 'normalize'],
        ignore_extra=True,
    )
    return out_params


def handle_RandomForestRegressor(options):
    out_params = convert_params(
        options.get('params', {}),
        ints=[
            'random_state',
            'n_estimators',
            'max_depth',
            'min_samples_split',
            'max_leaf_nodes',
        ],
        strs=['max_features'],
        ignore_extra=True,
    )

    if 'max_depth' not in out_params:
        out_params.setdefault('max_leaf_nodes', 2000)

    if 'max_features' in out_params:
        # Handle None case
        if out_params['max_features'].lower() == "none":
            out_params['max_features'] = None
        else:
            # EAFP... convert max_features to int if it is a number.
            try:
                out_params['max_features'] = float(out_params['max_features'])
                max_features_int = int(out_params['max_features'])
                if out_params['max_features'] == max_features_int:
                    out_params['max_features'] = max_features_int
            except:
                pass
    return out_params


class SystemIdentification(BaseAlgo):

    def __init__(self, options):
        self.handle_options(options)

        inital_params = convert_params(
            options.get('params', {}), 
            strs=['model', 'lags', 'time_field'],
            ignore_extra=True,
        )

        self.model_name = inital_params.pop('model', 'LinearRegression')
        self.lags = inital_params.pop('lags', '1h').split(',')
        self.time_field = inital_params.pop('time_field', TIME_FIELD)

        if self.model_name == 'RandomForestRegressor':
            output_params = handle_RandomForestRegressor(options)
            model = RandomForestRegressor(**output_params)
        else:
            output_params = handle_LinearRegression(options)
            model = LinearRegression(**output_params)

        self.estimator = MultiOutputRegressor(model)



    def handle_options(self, options):
        """Utility to ensure there are both target and feature variables"""
        self.feature_variables = options.get('feature_variables', [])
        self.target_variables = options.get('target_variable', [])
        if (
            self.target_variables < 1 or self.feature_variables == 0
        ):
            raise RuntimeError('Syntax error: expected "<target> ... FROM <field> ..."')

    def fit(self, df, options):
        df_now = df.copy()

        self.useable_variables = copy.copy(self.feature_variables)
        if self.time_field not in self.feature_variables:
            self.feature_variables.append(self.time_field)

        df_now[self.time_field] = pd.to_datetime(df_now[self.time_field])

        # check timestamp continuity
        # check missing data

        df_all = [df_now]
        features_all = []

        for lag in self.lags:
            lag_handler = LagHandler(lag)
            df_lagged, lagged_fields = lag_handler.process_df(df_now, self.useable_variables, self.target_variables, self.time_field)
            df_all.append(df_lagged)
            features_all += lagged_fields

        df_training = functools.reduce(
            lambda left, right: pd.merge(left, right, left_on=self.time_field, right_on=self.time_field, how='inner'), 
            df_all
        )
        
        X = df_training[features_all].values
        y = df_training[self.target_variables].values

        self.estimator.fit(X, y)


    def apply(self, df, options):
        df_new = df.copy(deep=True)

        df_new[self.time_field] = pd.to_datetime(df_new[self.time_field])

        df_all = []
        features_all = []

        for lag in self.lags:
            lag_handler = LagHandler(lag)
            df_lagged, lagged_fields = lag_handler.process_df(df_new, self.useable_variables, self.target_variables, self.time_field)
            df_all.append(df_lagged)
            features_all += lagged_fields

        df_prediction = functools.reduce(
            lambda left, right: pd.merge(left, right, left_on=self.time_field, right_on=self.time_field, how='inner'), 
            df_all
        )

        X = df_prediction[features_all].values

        y_pred = self.estimator.predict(X)
        
        output_names = ['predicted({})'.format(field) for field in self.target_variables]
        df_pred = pd.DataFrame(data=y_pred, columns=output_names, index=df_prediction.index)
        df_pred[self.time_field] = df_prediction[self.time_field]
        df_output = pd.merge(df_new, df_pred, left_on=self.time_field, right_on=self.time_field, how='outer', sort=True)

        return df_output

    def _summary(self):
        pass

    @staticmethod
    def _register_codecs():
        from codec.codecs import SimpleObjectCodec, TreeCodec
        codecs_manager.add_codec(
            "algos_forecast.SystemIdentification", "SystemIdentification", SimpleObjectCodec
        )
        codecs_manager.add_codec(
            "sklearn.multioutput", "MultiOutputRegressor", SimpleObjectCodec
        )
        codecs_manager.add_codec(
            "sklearn.linear_model.base", "LinearRegression", SimpleObjectCodec
        )
        codecs_manager.add_codec(
            'sklearn.ensemble.forest', 'RandomForestRegressor', SimpleObjectCodec
        )
        codecs_manager.add_codec(
            'sklearn.tree.tree', 'DecisionTreeRegressor', SimpleObjectCodec
        )
        codecs_manager.add_codec('sklearn.tree._tree', 'Tree', TreeCodec)

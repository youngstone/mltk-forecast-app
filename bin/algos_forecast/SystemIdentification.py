#!/usr/bin/env python
import functools
import re
import copy
from datetime import timedelta

import pandas as pd
import numpy as np

from scipy.stats import norm
from pandas.core.tools.datetimes import _guess_datetime_format_for_array
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from codec import codecs_manager
from base import BaseAlgo, ClassifierMixin
from util.param_util import convert_params
from util import df_util

import cexc

logger = cexc.get_logger(__name__)

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


class LagHandler(object):
    def __init__(self, lag):
        self.lag = lag

    @staticmethod
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

    def process_df(self, df, feature_fields, time_field):
        df_lagged = df[feature_fields].copy(deep=True)
        time_delta = self.convert_lag(self.lag)
        df_lagged[time_field] = df_lagged[time_field] + time_delta
        relevant_fields = [field for field in feature_fields if field != time_field]
        lagged_fields = [col + '@' + self.lag for col in relevant_fields]
        df_lagged.rename(columns={col: new_col for col, new_col in zip(relevant_fields, lagged_fields)}, inplace=True)
        return df_lagged, lagged_fields


class ModelHelper(object):
    def __init__(self, model_name):
        self.model_name = model_name
        if self.model_name == 'LinearRegression':
            self.handler = LinearRegressionHandler()
        elif self.model_name == 'RandomForestRegressor':
            self.handler = RandomForestRegressorHandler()
        else:
            raise RuntimeError('model={} is not supported.'.format(model_name))

    def handle_params(self, options):
        return self.handler.convert_options(options)

    def handle_model(self, out_params):
        return self.handler.initialize_model(out_params)

    def handle_confidence_interval(self, model, X, percentile):
        return self.handler.generate_confidence_interval(model, X, percentile)

    def handle_summary(self, estimator):
        return self.handler.generate_summary(estimator)


class RandomForestRegressorHandler(object):

    @staticmethod
    def convert_options(options):
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

    @staticmethod
    def initialize_model(out_params):
        return RandomForestRegressor(**out_params)

    @staticmethod
    def generate_confidence_interval(model, X, percentile=95):
        """
        Reference: https://blog.datadive.net/prediction-intervals-for-random-forests/
        """
        err_down = []
        err_up = []
        for i in range(len(X)):
            preds = []
            for est in model.estimators_:
                preds.append(est.predict(X[i].reshape(1, -1))[0])
            err_down.append(np.percentile(preds, (100 - percentile) / 2.))
            err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
        return err_down, err_up

    @staticmethod
    def generate_summary(estimator):
        df = pd.DataFrame(
            {'feature': estimator.feature_names, 'importance': estimator.feature_importances_.ravel()}
        )
        return df

class LinearRegressionHandler(object):

    @staticmethod
    def convert_options(options):
        out_params = convert_params(
            options.get('params', {}),
            bools=['fit_intercept', 'normalize'],
            ignore_extra=True,
        )
        return out_params

    @staticmethod
    def initialize_model(out_params):
        return LinearRegression(**out_params)

    @staticmethod
    def generate_confidence_interval(model, X, percentile):
        stderr = model.y_stderr
        y_pred = model.predict(X)
        scale = norm.ppf(percentile / 100.)
        upper = y_pred + scale * stderr
        lower = y_pred - scale * stderr
        return lower, upper

    @staticmethod
    def generate_summary(estimator):
        df = pd.DataFrame(
            {'feature': estimator.feature_names, 'coefficient': estimator.coef_.ravel()}
        )
        idf = pd.DataFrame(
            {'feature': ['_intercept'], 'coefficient': [estimator.intercept_]}
        )
        return pd.concat([df, idf])


class SystemIdentification(BaseAlgo):
    def __init__(self, options):
        self.handle_options(options)

        inital_params = convert_params(
            options.get('params', {}),
            strs=['model', 'lags', 'time_field'],
            ints=['confidence_interval'],
            ignore_extra=True,
        )
        self.model_name = inital_params.get('model')
        self.lags = inital_params.get('lags', '').split(',')
        self.time_field = inital_params.get('time_field', TIME_FIELD)
        self.confidence_interval = inital_params.get('confidence_interval', 95)

        if self.time_field not in self.feature_variables:
            self.feature_variables.append(self.time_field)

        self.model_helper = ModelHelper(self.model_name)

        out_params = self.model_helper.handle_params(options)
        model = self.model_helper.handle_model(out_params)

        self.estimator = MultiOutputRegressor(model)

    def handle_options(self, options):
        """Utility to ensure there are both target and feature variables"""
        self.feature_variables = options.get('feature_variables', [])
        self.target_variables = options.get('target_variable', [])
        if len(self.target_variables) < 1 or len(self.feature_variables) == 0:
            raise RuntimeError('Syntax error: expected "<target> ... FROM <field> ..."')
        for field in self.target_variables:
            if field not in self.feature_variables:
                self.feature_variables.append(field)

    def fit(self, df, options):
        df_now = df.copy(deep=True)

        df_now[self.time_field] = pd.to_datetime(df_now[self.time_field], unit="s")

        # check timestamp continuity
        # check missing data

        df_all = [df_now]
        features_all = []

        for lag in self.lags:
            lag_handler = LagHandler(lag)
            df_lagged, lagged_fields = lag_handler.process_df(df_now, self.feature_variables, self.time_field)
            df_all.append(df_lagged)
            features_all += lagged_fields

        df_training = functools.reduce(
            lambda left, right: pd.merge(left, right, left_on=self.time_field, right_on=self.time_field, how='inner'),
            df_all
        )

        X = df_training[features_all].values
        y = df_training[self.target_variables].values

        self.estimator.fit(X, y)
        y_pred = self.estimator.predict(X)
        y_stderr = np.sqrt(np.sum((y_pred - y)**2, axis=0) / (len(y) - 2))

        # information for post-process
        for i, estimator in enumerate(self.estimator.estimators_):
            estimator.feature_names = features_all
            estimator.y_stderr = y_stderr[i]

    def apply(self, df, options):
        df_new = df.copy(deep=True)
        df_new[self.time_field] = pd.to_datetime(df_new[self.time_field], unit="s")

        df_all = []
        features_all = []

        for lag in self.lags:
            lag_handler = LagHandler(lag)
            df_lagged, lagged_fields = lag_handler.process_df(df_new, self.feature_variables, self.time_field)
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

        for target, estimator in zip(self.target_variables, self.estimator.estimators_):
            lower, upper = self.model_helper.handle_confidence_interval(estimator, X, self.confidence_interval)
            df_pred['upper{}(predicted({}))'.format(self.confidence_interval, target)] = upper
            df_pred['lower{}(predicted({}))'.format(self.confidence_interval, target)] = lower

        df_pred[self.time_field] = df_prediction[self.time_field]
        df_output = pd.merge(df_new, df_pred, left_on=self.time_field, right_on=self.time_field, how='outer', sort=True)

        return df_output

    def summary(self, options):
        if len(options) != 2:  # only model name and mlspl_limits
            raise RuntimeError(
                '"%s" models do not take options for summarization' % self.__class__.__name__
        )
        df_summary_all = []
        for target, estimator in zip(self.target_variables, self.estimator.estimators_):
            # estimator.feature_names = self.estimator.feature_names
            df_summary = self.model_helper.handle_summary(estimator)
            df_summary['target'] = target
            df_summary_all.append(df_summary)
        return pd.concat(df_summary_all, ignore_index=True)

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec, TreeCodec
        codecs_manager.add_codec("algos_forecast.SystemIdentification", "ModelHelper", SimpleObjectCodec)
        codecs_manager.add_codec("algos_forecast.SystemIdentification", "LinearRegressionHandler", SimpleObjectCodec)
        codecs_manager.add_codec("algos_forecast.SystemIdentification", "RandomForestRegressorHandler", SimpleObjectCodec)
        codecs_manager.add_codec("algos_forecast.SystemIdentification", "SystemIdentification", SimpleObjectCodec)
        codecs_manager.add_codec("sklearn.multioutput", "MultiOutputRegressor", SimpleObjectCodec)
        codecs_manager.add_codec("sklearn.linear_model.base", "LinearRegression", SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.ensemble.forest', 'RandomForestRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree.tree', 'DecisionTreeRegressor', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.tree._tree', 'Tree', TreeCodec)

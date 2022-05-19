import os
import psycopg2
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import yaml
import re
import datetime as dt

import plotly.express as px
import plotly.graph_objects as go

import statistics
from scipy.interpolate import UnivariateSpline

with open(os.path.join(os.path.dirname(__file__), 'rpm-cfg.yml'), 'rb') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

# Get today's date
def _get_date():
    return dt.datetime.now().date()

def _get_date_str():
    return _get_date().strftime('%Y%m%d')

today = _get_date_str()

class RPMDB:
    query_dir = os.path.join(os.path.dirname(__file__), cfg['query_dir'])

    queries = [f for f in os.listdir(query_dir) if '.sql' in f]

    date_cols = cfg['date_cols']

    def __init__(self, host, user, password, port=5432, autorefresh=None):
        self.conn_str = self._conn_str(host, user, password, port)

        self.autorefresh = autorefresh

        self.method_map = self._method_map()

        self.refresh()

    def _method_map(self):
        return {
            'billing': self.get_billing_data_clean,
            'usage': self.get_usage_data_clean,
            'measures': self.get_meas_data
        }

    def _conn_str(self, host, user, password, port=5432):
        stem = cfg['conn_str']

        return stem.format(
            host=host,
            user=user,
            password=password,
            port=port
        )

    def _connector(self):
        return psycopg2.connect(self.conn_str)

    def refresh(self):
        for q in self.queries:
            with open(os.path.join(self.query_dir, q), 'r') as file:
                query = file.read()

            engine = create_engine('postgresql+psycopg2://', creator=self._connector)

            dat = pd.read_sql(
                    sql = query,
                    con = engine,
                    parse_dates = [d for d in self.date_cols if d in query]
                )

            self.__dict__[q.replace('.sql', '')] = dat

        self.refresh_time = dt.datetime.now()

    def last_refresh(self):
        return dt.datetime.utcnow() - self.refresh_time

    @staticmethod
    def _get_query(q):
        with open(os.path.join(self.query_dir, q + '.sql'), 'rb') as file:
            return file.read()

    def get_billing_data(self, refresh=True):
        if refresh:
            self.refresh()

        return self.billing

    def get_usage_data(self, refresh=True):
        if refresh:
            self.refresh()

        return self.usage

    def get_meas_data(self, refresh=True):
        if refresh:
            self.refresh()

        return self.measures

    def get_members(self, refresh=True):
        if refresh:
            self.refresh()

        return self.members

    def get_billing_data_clean(self, refresh=True):
        bd = self.get_billing_data(refresh=refresh)

        bd['mrn'] = np.where(
            bd.extern_id.str.contains('^\d+$', regex=True),
            bd.extern_id.str.pad(8, 'left', '0'),
            None
        )

        bd = bd[bd.last_name != 'Test'][['full_name', 'mrn', 'email', 'billing_period', 'start_period', 'end_period', 'measurements_per_period', 'setup_code', 'high_use_code']]\
            .replace(
                {
                    'setup_code': {0: 'Ineligible', 1: 'Eligible'},
                    'high_use_code': {0: 'Ineligible', 1: 'Eligible'}
                }
            )\
            .rename(columns={
                'full_name': 'Name',
                'mrn': 'MRN',
                'email': 'Email',
                'billing_period': 'Billing Period',
                'start_period': 'Period Start',
                'end_period': 'Period End',
                'measurements_per_period': 'Number of Measurements in Period',
                'setup_code': 'Setup Code (CPT99453)',
                'high_use_code': 'High Use Code (CPT99454)'
            })

        return bd

    def get_usage_data_clean(self, refresh=True):
        ud = self.get_usage_data(refresh=refresh)

        ud['mrn'] = np.where(
            ud.extern_id.str.contains('^\d+$', regex=True),
            ud.extern_id.str.pad(8, 'left', '0'),
            None
        )

        ud = ud[ud.last_name != 'Test'][['full_name', 'mrn', 'email', 'last_measured_at', 'days_since_last_measurement']]\
            .rename(
                columns={
                    'full_name': 'Name',
                    'mrn': 'MRN',
                    'email': 'Email',
                    'last_measured_at': 'Last Measurement',
                    'days_since_last_measurement': 'Days Since the Last Measurement'
                }
            )

        return ud

    def safe_agg(x, agg):
        if agg == 'mean':
            return x.dropna().mean()
        elif agg == 'median':
            return x.dropna().median()
        elif agg == 'stdev':
            return statistics.stdev(x.dropna()) if len(x.dropna()) > 1 else 0


    def get_stream(self, measure, member_id, tz='America/New_York', refresh=True, time_aggregation=None):
        if type(member_id) != list:
            member_id = [member_id]

        md_dat = self.get_meas_data(refresh=refresh)

        md_dat['measured_at'] = md_dat['measured_at'].dt.tz_convert(tz)

        md_dat = md_dat\
            .sort_values('measured_at')

        pt_data = md_dat[
            (md_dat.measure == measure) &\
            (md_dat.member_id.isin(member_id))
        ].copy()

        if time_aggregation:
            if time_aggregation == 'day':
                pt_data['date_col'] = pt_data.measure_day_et

                pt_data['date_col_numeric'] = pt_data.date_col.values.astype(float)

            pt_data = pt_data\
                .groupby(['measure', 'date_col', 'date_col_numeric'])\
                .agg(
                    value_numeric=pd.NamedAgg('value_numeric', lambda x: safe_agg(x, agg='median')),
                    value_numeric_mean=pd.NamedAgg('value_numeric', lambda x: safe_agg(x, agg='mean')),
                    value_numeric_sd=pd.NamedAgg('value_numeric', lambda x: safe_agg(x, 'stdev')),
                    delta_from_last=pd.NamedAgg('delta_from_last', lambda x: safe_agg(x, 'median')),
                    delta_from_last_mean=pd.NamedAgg('delta_from_last', lambda x: safe_agg(x, 'mean')),
                    delta_from_last_sd=pd.NamedAgg('delta_from_last', lambda x: safe_agg(x, 'stdev'))
                )\
                .reset_index()
        else:
            pt_data['date_col'] = pt_data.measured_at

            pt_data['date_col_numeric'] = pt_data.date_col.values.astype(float)

        arr1 = pt_data.value_numeric

        # finding the 1st quartile
        q1 = np.quantile(arr1, 0.25)

        # finding the 3rd quartile
        q3 = np.quantile(arr1, 0.75)
        med = np.median(arr1)

        # finding the iqr region
        iqr = q3-q1

        # finding upper and lower whiskers
        upper_bound = q3+(1.5*iqr)
        lower_bound = q1-(1.5*iqr)

        # outliers
        pt_data['outlier'] = np.where(
            (pt_data.value_numeric <= lower_bound) |\
            (pt_data.value_numeric >= upper_bound),
            1, 0
        )

        inliers = pt_data[pt_data.outlier == 0]
        outliers = pt_data[pt_data.outlier == 1]

        return {
            'full': pt_data.copy(),
            'inliers': inliers.copy(),
            'outlier': outliers.copy()
        }

    def plot(self, data_dict=None, measure=None, member_id=None, smoothing_factor=3600 * 24, layout={}, tz='America/New_York', refresh=refresh, time_aggregation=None):
        if data_dict is None:
            data_dict = self.get_stream(measure, member_id, tz=tz, refresh=refresh, time_aggregation=time_aggregation)

        inliers = data_dict['inliers']
        outliers = data_dict['outlier']
        full = data_dict['full']

        spl = UnivariateSpline(inliers.date_col_numeric, inliers.value_numeric)
        spl.set_smoothing_factor(smoothing_factor)

        colors = {
            'inliers': 'rgba(45,112,142,0.5)',
            'outliers': 'rgba({}, {}, {}, 0.5)'.format(*(255 * i for i in  (0.698, 0.133, 0.133))),
            'smoothed': 'rgba(45,112,142,1)'
        }

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=inliers.date_col,
                y=inliers.value_numeric,
                mode='markers',
                marker={
                    'color': colors['inliers']
                },
                name='Inliers'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=outliers.date_col,
                y=outliers.value_numeric,
                mode='markers',
                marker={
                    'color': colors['outliers'],
                    'symbol': 'x'
                },
                name='Outliers'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=inliers.date_col,
                y=spl(inliers.date_col_numeric),
                mode='lines',
                line=dict(
                    color=colors['smoothed'],
                    shape='spline',
                    width=2
                ),
                name='Smoothed'
            )
        )

        latest_date = full.date_col.max().strftime('%m/%d/%Y %H:%M:%S')
        measure_title = measure.title()

        default_layout = dict(
            title=f'{measure_title} measurement stream<br><sup>Latest measurement: {latest_date}</sup>',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis={
                'title': f'{measure_title}'
            },
            xaxis={
                'title': None
            }
        )

        layout = {**default_layout, **layout}

        fig.update_layout(
            **layout
        )

        return fig

    # Ensure data path
    @staticmethod
    def _ensure_path(fp):
        if not os.path.exists(fp):
            os.mkdir(fp)

    # Get a filename
    @staticmethod
    def _get_fp(data_dir, bases, format='csv'):
        if 'xls' in format:
            joined_base = '_'.join(bases)

            return os.path.join(data_dir, f'{joined_base}_{today}.{format}')

        return {b: os.path.join(data_dir, f'{b}.{format}') for b in bases}

    # Write the query data to a file in the data_dir
    @staticmethod
    def _write_data(dfs, fps, format):
        if re.search('csv', format):
            for k in dfs.keys():
                dfs[k].to_csv(fps[k], index=False)
        elif re.search('xlsx?', format):
            ew = pd.ExcelWriter(fps)

            out_dict = dfs

            for df in dfs.values():
                tz_awares = [_ for _, d in df.dtypes.iteritems() if type(d) == pd.DatetimeTZDtype]

                for t in tz_awares:
                    df[t] = df[t].dt.tz_convert(None)

            for k, d in out_dict.items():
                d.to_excel(ew, sheet_name=k, startrow=0, index=False, header=True)

            ew.save()
        else:
            raise Exception('Unsupported format')

    # Write data to a path
    def write_data(self, endpoints, data_dir='data', query_params='', format='csv'):
        endpoints = [endpoints] if type(endpoints) != list else endpoints

        fps = self._get_fp(data_dir, endpoints, format)

        dats = {e: self.method_map[e]() for e in endpoints}

        self._ensure_path(data_dir)

        self._write_data(dats, fps, format)

        return list(fps.values()) if type(fps) == dict else fps

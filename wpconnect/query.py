import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import os
from pyodbc import Error, ProgrammingError, DatabaseError
from sqlalchemy.exc import ResourceClosedError
from sqlalchemy import text
from .connect import Connect
from .settings import Settings
import warnings

from github import Github
import re
import requests
import pkgutil

import time
import datetime as dt
import pytz

import base64

settings = Settings()

def get_rate_limits(g):
    rate_remaining, rate_limit = g.rate_limiting

    reset_time = g.rate_limiting_resettime

    return rate_remaining, rate_limit, reset_time

def rate_aware(func):
    """Decorator for rate-aware Github API calls
    """
    def wrapper(self, *args, **kwargs):
        rate_remaining, rate_limit, reset_time = get_rate_limits(self.g)

        if rate_remaining < 1:
            local_reset = dt.datetime\
                .fromtimestamp(reset_time)\
                .astimezone(
                    pytz.timezone('America/New_York')
                )

            local_now = dt.datetime\
                .now(pytz.UTC)\
                .astimezone(
                    pytz.timezone('America/New_York')
                )

            wait_time = (local_reset - local_now).total_seconds()

            if wait_time > 0:
                time.sleep(wait_time)

        res = func(self, *args, **kwargs)

        rate_remaining, rate_limit, reset_time = get_rate_limits(self.g)

        remaining_ratio = rate_remaining / rate_limit

        if remaining_ratio < settings.GITHUB_RATE_WARNING_THRESHOLD:
            warnings.warn(f'{remaining_ratio:.1%} of rate limit remaining. Execution will pause if rate limit is reached')

        return res

    return wrapper

class Query:
    g = None
    r = None
    cfs = {}

    def __init__(
        self,
        connection_type=None,
        environ=None,
        server=settings.WPH_SERVER,
        database=settings.WPH_DATABASE,
        port=None,
        username=None,
        password=None,
        repo=None,
        trusted_connection=True,
        make_password_safe=True,
    ):
        self.connection = Connect(connection_type, environ, server, database, port, username, password, trusted_connection, make_password_safe)
        self.conn = self.connection.conn
        self.query_libs = ['.']

        self.repo_config = repo is not None

        if self.repo_config:
            self.repo = repo

            self.repo_duplicates = []

            self.configure_repo()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.connection.close()

    def add_query_libs(
        self,
        query_libs: list
    ):
        query_libs = [query_libs] if type(query_libs) != list else query_libs

        for l in query_libs:
            if l not in self.query_libs:
                self.query_libs.append(l)

    @rate_aware
    def _get_dirs_at_level(
        self,
        d
    ):
        level_dirs = []

        # Identify directories at this level
        for cf in self.r.get_contents(d):
            path = cf.path

            if cf.type == 'dir':
                level_dirs.append(cf.path)
            elif '.sql' in cf.path:
                if cf.name in self.cfs.keys():
                    self.repo_duplicates.append(cf.name)

                self.cfs[cf.name] = cf.sha

        return level_dirs

    @rate_aware
    def get_cfs(self):
        # Get all directories in the repo
        dirs = ['.']
        scanned_dirs = []

        needs_scanning = [d for d in dirs if d not in scanned_dirs]

        while len(needs_scanning) > 0:
            for d in dirs:
                level_dirs = self._get_dirs_at_level(d)

                dirs += level_dirs

                scanned_dirs.append(d)

                needs_scanning = [d for d in dirs if d not in scanned_dirs]

        if len(self.repo_duplicates) > 0:
            joined_dups = ', '.join(self.repo_duplicates)

            warnings.warn(f'Duplicated queries found: {joined_dups}')

    def configure_repo(
        self
    ):
        needed_keys = ['access_token', 'username', 'repo']

        missing_keys = [k for k in needed_keys if k not in self.repo.keys()]

        joined_missing = ', '.join(missing_keys)

        if len(missing_keys) > 0:
            raise UserError(f'Missing keys in query argument: {joined_missing}')

        repo_name = '%s/%s' % (self.repo['username'], self.repo['repo'])

        self.g = Github(self.repo['access_token'])

        self.r = self.g.get_repo(repo_name)

        self.get_cfs()

    @rate_aware
    def _get_repo_query(
        self,
        filename
    ):
        res = self.r.get_git_blob(self.cfs[filename])

        raw_data = res.raw_data['content']

        decoded = base64.b64decode(raw_data).decode()

        return decoded

    def import_sql(
        self,
        filename,
    ):
        if self.repo_config:
            query = self._get_repo_query(filename)
        else:
            # First try to import from package data
            try:
                query = pkgutil.get_data(__name__, os.path.join('queries', filename)).decode('ascii')
            except FileNotFoundError:
                query = None

            if query is None:
                for l in self.query_libs:
                    if filename in os.listdir(l):
                        with open(os.path.join(l, filename)) as file:
                            query = file.read()

        return query

    @staticmethod
    def _replace_dates(
        df : pd.DataFrame,
        dtypes : pd.DataFrame
    ):
        for c in dtypes:
            if any([is_datetime(r) for r in dtypes[c]]):
                df[c] = pd.to_datetime(df[c])

        return df

    # TODO: PARAMS WILL NOT WORK WITH SQLALCHEMY/ORACLE
    def execute_query(
        self,
        query: str,
        params: list = None,
        return_type: str = settings.DEFAULT_RETURN,
        chunksize: int = None,
    ):
        if query:
            if params:
                params = [params] if type(params) != list else params

            if return_type is None:
                try:
                    qr = self.conn.execute(text(query))
                    # TODO: check if ddl. if not allow params
                    # TODO: check if conn is from sqlalchemy. if not, use self.conn.cursor()
                except pd.io.sql.DatabaseError as err:
                    warnings.warn(str(err), UserWarning)
                    return

                return
            elif return_type == 'DataFrame':
                try:
                    db_res = pd.read_sql(text(query), self.conn, params=params, chunksize=chunksize)

                    if chunksize:
                        dtypes = []
                        frames = []
                        for chunk in db_res:
                            dtypes.append(chunk.dtypes)
                            frames.append(chunk)

                        try:
                            dtypes = pd.DataFrame(dtypes)
                            res = self._replace_dates(
                                pd.concat(frames),
                                dtypes
                            )

                            del dtypes
                        except Exception as err:
                            res = pd.concat(frames)

                            warnings.warn('Could not fix dates (likely due to duplicated columns in the original query)')
                    else:
                        res = db_res

                except pd.io.sql.DatabaseError as err:
                    warnings.warn(str(err), UserWarning)
                    return
                except ResourceClosedError as err:
                    if 'does not return rows' in str(err):
                        raise ValueError('The query did not return any rows. If this is expected by design, please call execute_query(..., return_type=None)')
                        return
                    else:
                        raise err
                except AttributeError as err:
                    if 'NoneType' in str(err):
                        raise ValueError('The query did not return any rows. If this is expected by design, please call execute_query(..., return_type=None)')
                        return
                    else:
                        raise err
            else:
                try:
                    with self.conn.cursor() as cursor:
                        qr = cursor.execute(text(query), params=params)
                        res = qr.fetchall()
                except pd.io.sql.DatabaseError as err:
                    warnings.warn(str(err), UserWarning)
                    return
                except AttributeError as err:
                    qr = self.conn.execute(text(query))
                    res = qr.fetchall()

            self.most_recent_query = query
            self.most_recent_params = params
        else:
            warnings.warn('No query provided', UserWarning)
            return

        return res

    def execute_imported(
        self,
        query_file: str,
        params: list = None,
        return_type: str = settings.DEFAULT_RETURN,
        chunksize: int = None,
    ):
        query = self.import_sql(query_file)

        return self.execute_query(query, params=params, return_type=return_type, chunksize=chunksize)

    def list_tables(
        self,
        return_type: str = settings.DEFAULT_RETURN,
    ):
        return self.execute_imported('list_tables.sql')

    def list_objects(
        self,
        return_type: str = settings.DEFAULT_RETURN,
    ):
        return self.execute_imported('list_objects.sql')


    def list_columns(
        self,
        return_type: str = settings.DEFAULT_RETURN,
    ):
        return self.execute_imported('list_columns.sql')

    def to_sql(
        self,
        df,
        **kwargs
    ):
        try:
            conn = self.connection.engine
        except AttributeError as err:
            conn = self.conn

            print(str(err))

        return df.to_sql(
            con=conn,
            **kwargs
        )

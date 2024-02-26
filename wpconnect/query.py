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

from sqlparse.tokens import *
import sqlparse

import gc

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
                if self.github_rate_action == settings.GITHUB_RATE_PAUSE:
                    wait_time_minutes = wait_time // 60

                    warnings.warn(f'Rate limit exceeded. Waiting {wait_time_minutes} minutes')

                    time.sleep(wait_time)
                elif self.github_rate_action == settings.GITHUB_RATE_KILL:
                    pass

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
        github_rate_action=settings.GITHUB_RATE_PAUSE
    ):
        self.connection = Connect(connection_type, environ, server, database, port, username, password, trusted_connection, make_password_safe)
        self.conn = self.connection.conn
        self.query_libs = ['.']

        self.github_rate_action = github_rate_action

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

    @classmethod
    def get_cfs(cls):
        return cls.cfs

    @classmethod
    def set_cfs(cls, cfs):
        cls.cfs = cfs

    @rate_aware
    def _get_dirs_at_level(
        self,
        d
    ):
        level_dirs = []
        new_cfs = {}

        # Identify directories at this level
        for cf in self.r.get_contents(d):
            path = cf.path

            if cf.type == 'dir':
                level_dirs.append(cf.path)
            elif '.sql' in cf.path:
                new_cfs[cf.name] = cf.sha

        return level_dirs, new_cfs

    @rate_aware
    def scan_cfs(self):
        # Get all directories in the repo
        dirs = ['.']
        scanned_dirs = []

        needs_scanning = [d for d in dirs if d not in scanned_dirs]

        cfs = {}
        repo_duplicates = []
        while len(needs_scanning) > 0:
            for d in dirs:
                level_dirs, new_cfs = self._get_dirs_at_level(d)

                new_dups = list(
                    set(list(new_cfs.keys())) &\
                    set(list(cfs.keys()))
                )

                if len(new_dups) > 0:
                    repo_duplicates += new_dups

                cfs = {
                    **cfs,
                    **new_cfs
                }

                dirs += level_dirs

                scanned_dirs.append(d)

                needs_scanning = [d for d in dirs if d not in scanned_dirs]

        if len(repo_duplicates) > 0:
            joined_dups = ', '.join(repo_duplicates)

            warnings.warn(f'Duplicated queries found: {joined_dups}')

        self.set_cfs(cfs)

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

        self.scan_cfs()

    @rate_aware
    def _get_repo_query(
        self,
        filename
    ):
        res = self.r.get_git_blob(self.cfs[filename])

        raw_data = res.raw_data['content']

        decoded = base64.b64decode(raw_data).decode(errors='replace')

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
        orig_df : pd.DataFrame,
        dtypes : pd.DataFrame
    ):
        df = orig_df.copy()

        for c in dtypes:
            if any([is_datetime(r) for r in dtypes[c]]):
                df[c] = pd.to_datetime(df[c])

        del orig_df

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

                            comb_fr = pd.concat(frames)

                            res = self._replace_dates(
                                comb_fr,
                                dtypes
                            )

                            del comb_fr
                            del dtypes
                        except Exception as err:
                            res = pd.concat(frames)

                            warnings.warn('Could not fix dates (likely due to duplicated columns in the original query)')

                        del frames
                    else:
                        res = db_res

                    del db_res

                    db_res = pd.DataFrame()

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

        gc.collect()

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

    def _filter_tokens(self, stmt):
        mode = ''
        res = {k:[] for k in ['select','from','cte','comments']}
        for token in stmt.tokens:
            # print(token.ttype)
            if isinstance(token, sqlparse.sql.Comment):
                res['comments'].append(str(token))
                mode = ''
            elif token.ttype == CTE:
                mode = 'cte'
            elif token.ttype == DML and str(token) == 'SELECT':
                mode = 'select'
            elif token.ttype == Keyword and (str(token)== 'FROM' or 'JOIN' in str(token)):
                mode = 'from'
            elif mode == 'select' and token.ttype==Wildcard:
                res[mode].append(token)
            elif mode != '' and token.ttype is None:                
                res[mode].extend(token)
                mode = ''    
        return res

    def _parse_query(self, stmt):
        
        res = self._filter_tokens(stmt)
        res['comments'] = ' '.join(res['comments'])
        cte_list = self._parse_identifiers(res['cte'],cte=True)
        cte = {}
        for d in cte_list:
            cte.update(d)
        res['cte'] = cte

        columns = self._parse_identifiers(res['select'])
        tables  = self._parse_identifiers(res['from'])
     
        res['from'] = tables
        
        if '*' not in ''.join(columns):
            res['select'] = columns
        else:
            x = [n.split('.')[0].upper() for n in columns if '.' in n]
            if len(x) > 0:
                tables = x
                
            for t in tables:
                if t in cte:
                    columns.extend(cte[t]['select'])
                else:
                    columns.append(f'From Table: {t}')
            columns = [c for c in columns if '*' not in c]
        for c in columns:
            if 'From Table' in c and c.split(': ')[-1] in cte:
                columns.remove(c)
                columns.extend(cte[c.split(': ')[-1]]['select'])
            
        res['select'] = columns
        res = {k:v for k, v in res.items() if len(v) > 0}
        self.struct = res
        return res

    def _parse_identifiers(self, tokens, cte=False):
    
        name  = []
        temp = {}
        res = []
        for token in tokens:

            if isinstance(token, sqlparse.sql.IdentifierList):
                res.append([i.get_name().upper() for i in token.get_identifiers()]) 
            elif token.ttype == Name:
                if len(name) > 0 and name[-1][-1] == '.':
                    name[-1] = name[-1] + str(token).upper()                
                else:
                    name.append(str(token).upper())
                
            elif isinstance(token, sqlparse.sql.Parenthesis):
                temp = self._parse_query(token)

                if cte:
                    res.append({name[-1]:temp})
                    temp = {}
                 
            elif cte and isinstance(token, sqlparse.sql.Identifier): 

                dd = [t for t in token if t.ttype is None][0]
                res.extend(name)
                name = []
                res.append({token.get_name().upper(): self._parse_query(dd)})
            elif not cte and isinstance(token, sqlparse.sql.Identifier) and not '@' in str(token): 
                nn = token.get_name().upper()
                if '*' in nn:
                    nn = str(token) 
                if temp:
                    res.append({nn: temp})
                    temp = {}
                elif len(name) > 0:
                    name = ' '.join(name).replace('. ','.').split(' ')
                    res.extend(name)
                    name = []
                else:
                    res.append(nn)
            elif str(token) == '.':
                name[-1] = name[-1] + '.'
                while len(name) > 1:
                    res.append(name.pop(0))
                
            elif '@' in str(token):
                if len(name) > 0:
                    name[-1] = name[-1] + str(token)  
                    while len(name) > 1:
                        res.append(name.pop(0))
                else:
                    res.append(str(token))
            elif token.ttype == Wildcard:
                if len(name) > 1:
                    name[-1] = name[-1] + '*'
                    res.append(name.pop(-1))
                else:
                    res.append('*')

        if len(name) > 0:
        
            name = ' '.join(name).replace('. ','.').split(' ')
            if temp:
                res.append([{name.pop(-1):temp}])
            res.extend(name)
            name = []
        return res

    def parse_query(self, filename: str):
        query = self.import_sql(filename)
        stmt = sqlparse.parse(query)[0]
        return = self._parse_query(stmt)
       
    def get_comments(self, query_fn):
        c = self.parse_query(query_fn)['comments']
        
        sec_starts = [i.span()[0] for i in re.finditer(r'[^\:\n]+(?=\:)', c)]
        
        if len(sec_starts) > 0:
            ln = []
            for i in range(len(sec_starts)-1):
                ln.append(c[sec_starts[i]:sec_starts[i+1]])
            ln.append(c[sec_starts[-1]:].split('*')[0])
            
            header = {l.split(':')[0]:l.split(':')[-1].strip(' \n')  for l in ln}
        else:
            header = {} 
        return header

    def get_columns(self, query_fn):
        return self.parse_query(query_fn)['select']
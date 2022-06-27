import pandas as pd
import os
from pyodbc import Error, ProgrammingError, DatabaseError
from sqlalchemy.exc import ResourceClosedError
from .connect import Connect
from .settings import Settings
import warnings

from github import Github
import re
import requests
import pkgutil


settings = Settings()

class Query:
    def __init__(
        self,
        connection_type=None,
        environ=None,
        server=settings.WPH_SERVER,
        database=settings.WPH_DATABASE,
        port=None,
        username=None,
        password=None,
        repo=None
    ):
        """Class used to request and access information from repositories.

        Args:
            connection_type: Select connection type. Defaults to None.
            environ: Select the environment. Defaults to None.
            server: Select desired database to use. Defaults to settings.WPH_SERVER.
            database: Slect which database to use. Defaults to settings.WPH_DATABASE.
            port: Slect port that will be in use. Defaults to None.
            username: Select username credentials. Defaults to None.
            password: Select password credentials. Defaults to None.
            repo: Select repository. Defaults to None.
        """
        self.connection = Connect(connection_type, environ, server, database, port, username, password)
        self.conn = self.connection.conn
        self.query_libs = ['.']

        self.repo_config = repo is not None

        if self.repo_config:
            self.repo = repo

            self.repo_duplicates = []

            self.configure_repo()

    def __enter__(self):
        """Magic method

        Returns:
            Function: Ability to implement objects using a with statement
        """
        return self

    def __exit__(self, type, value, tb):
        """Magic method

        Returns:
            Function: Ability to implement objects using a with statement
        """
        
        self.connection.close()

    def add_query_libs(
        self,
        query_libs: list
    ):
        """Function that lists requested quireis

        Args:
            query_libs (list): _description_
        """
        query_libs = [query_libs] if type(query_libs) != list else query_libs

        for l in query_libs:
            if l not in self.query_libs:
                self.query_libs.append(l)

    def _get_dirs_at_level(
        self,
        r,
        d
    ):
        """Function that describes and returns information based on directory levels

        Args:
            r (_type_): r level
            d (_type_): d level

        Returns:
            _type_: directory in list form based on the directory level selected
        """
        level_dirs = []

        # Identify directories at this level
        for cf in r.get_contents(d):
            path = cf.path

            if cf.type == 'dir':
                level_dirs.append(cf.path)

        return level_dirs

    def configure_repo(
        self
    ):
        """Describes requirements to Query information from repositiories (repo/s)

        Raises:
            UserError: If requirments aren't meant.
        """
        needed_keys = ['access_token', 'username', 'repo']

        missing_keys = [k for k in needed_keys if k not in self.repo.keys()]

        joined_missing = ', '.join(missing_keys)

        if len(missing_keys) > 0:
            raise UserError(f'Missing keys in query argument: {joined_missing}')

        repo_name = '%s/%s' % (self.repo['username'], self.repo['repo'])

        g = Github(self.repo['access_token'])

        r = g.get_repo(repo_name)

        # Get all directories in the repo
        dirs = ['.']
        scanned_dirs = []

        needs_scanning = [d for d in dirs if d not in scanned_dirs]

        while len(needs_scanning) > 0:
            for d in dirs:
                level_dirs = self._get_dirs_at_level(r, d)

                dirs += level_dirs

                scanned_dirs.append(d)

                needs_scanning = [d for d in dirs if d not in scanned_dirs]

        # Look through the repo dirs for sql queries
        cfs = {}

        for d in dirs:
            dir_contents = r.get_contents(d)

            for q in dir_contents:
                if '.sql' in q.path:
                    # TODO: evaluate if we can use basename instead of full
                    # base_name = re.search(re.compile('.+(?=\.)'), q.name)
                    #
                    # if base_name:
                    #     name = base_name.group(0)
                    # else:
                    #     name = q.name

                    name = q.name

                    if name in cfs.keys():
                        self.repo_duplicates.append(name)

                    cfs[name] = q.download_url

        if len(self.repo_duplicates) > 0:
            joined_dups = ', '.join(self.repo_duplicates)

            warnings.warn(f'Duplicated queries found: {joined_dups}')

        self.cfs = cfs

    def _get_repo_query(
        self,
        filename
    ):
        """_summary_

        Args:
            filename (_type_): _description_

        Returns:
            Function: Returns selected repository
        """
        headers = {'Authorization': 'token %s' % self.repo['access_token']}

        res = requests.get(self.cfs[filename], headers=headers)

        if res.status_code != 200:
            res.raise_for_status()

        return res.text

    def import_sql(
        self,
        filename,
    ):
        """Establishes ability to return sql query while querying the repositories

        Args:
            filename (_type_): _description_

        Returns:
            Function: Returns SQL query of the selected repo.
        """
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

    # TODO: PARAMS WILL NOT WORK WITH SQLALCHEMY/ORACLE
    def execute_query(
        self,
        query: str,
        params: list = None,
        return_type: str = settings.DEFAULT_RETURN,
    ):
        
        """Executes query under specified paramters

        Raises:
            ValueError: Query did not return any rows
            err: raised if aformentioned occured
            ValueError: Query did not return any rows
            err: raised if aformentioned occured

        Returns:
            Query: Query is exectued and returned 
        """
        if query:
            if params:
                params = [params] if type(params) != list else params

            if return_type is None:
                try:
                    qr = self.conn.execute(query)
                    # TODO: check if ddl. if not allow params
                    # TODO: check if conn is from sqlalchemy. if not, use self.conn.cursor()
                except pd.io.sql.DatabaseError as err:
                    warnings.warn(str(err), UserWarning)
                    return

                return
            elif return_type == 'DataFrame':
                try:
                    res = pd.read_sql(query, self.conn, params=params)
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
                        qr = cursor.execute(query, params=params)
                        res = qr.fetchall()
                except pd.io.sql.DatabaseError as err:
                    warnings.warn(str(err), UserWarning)
                    return
                except AttributeError as err:
                    qr = self.conn.execute(query)
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
        return_type: str = settings.DEFAULT_RETURN
    ):
        """Imported queiries can be returned

        Args:
            query_file (str): _description_
            params (list, optional): _description_. Defaults to None.
            return_type (str, optional): _description_. Defaults to settings.DEFAULT_RETURN.

        Returns:
            Query: imported queiries returned
        """
        query = self.import_sql(query_file)

        return self.execute_query(query, params=params, return_type=return_type)

    def list_tables(
        self,
        return_type: str = settings.DEFAULT_RETURN,
    ):
        """Returns a list of tables
        """
        return self.execute_imported('list_tables.sql')

    def list_objects(
        self,
        return_type: str = settings.DEFAULT_RETURN,
    ):
        """Returns a list of queried objects
        """
        return self.execute_imported('list_objects.sql')


    def list_columns(
        self,
        return_type: str = settings.DEFAULT_RETURN,
    ):
        """Returns the columns of the query
        """
        return self.execute_imported('list_columns.sql')

    def to_sql(
        self,
        df,
        **kwargs
    ):
        """Function that changes df format to SQL.
        """
        df.to_sql(
            con=self.conn,
            **kwargs
        )

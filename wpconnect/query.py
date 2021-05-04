import pandas as pd
import os
from pyodbc import Error, ProgrammingError, DatabaseError
from .connect import Connect
from .settings import Settings
import warnings

import pkgutil

settings = Settings()

class Query:
    def __init__(
        self,
        connection_type=None,
        server=settings.WPH_SERVER,
        database=settings.WPH_DATABASE,
        port=settings.WPH_PORT,
        username=None,
        password=None,
    ):
        self.connection = Connect(connection_type, server, database, port, username, password)
        self.conn = self.connection.conn
        self.query_libs = ['.']

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.connection.close()

    def add_query_libs(
        self,
        query_libs: list
    ):
        query_libs = [query_libs] if type(query_libs) != list else query_libs

        self.query_libs = self.query_libs + query_libs

    def import_sql(
        self,
        filename,
    ):
        # First try to import from package data
        query = pkgutil.get_data(__name__, os.path.join('queries', filename)).decode('ascii')

        if query is None:
            for l in self.query_libs:
                if filename in os.listdir(l):
                    with open(os.path.join(l, filename)) as file:
                        query = file.read()

        return query

    def execute_query(
        self,
        query: str,
        params: list = None,
        return_type: str = settings.DEFAULT_RETURN,
    ):
        if query:
            if params:
                params = [params] if type(params) != list else params

            if return_type == 'DataFrame':
                try:
                    res = pd.read_sql(query, self.conn, params=params)
                except pd.io.sql.DatabaseError as err:
                    warnings.warn(str(err), UserWarning)
                    return
            else:
                try:
                    with self.conn.cursor() as cursor:
                        qr = cursor.execute(query, params=params)
                        res = qr.fetchall()
                except pd.io.sql.DatabaseError as err:
                    warnings.warn(str(err), UserWarning)
                    return

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
        query = self.import_sql(query_file)

        return self.execute_query(query, params=params, return_type=return_type)

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
        df.to_sql(
            con=self.conn,
            **kwargs
        )

import pyodbc
import sqlalchemy as sa
import cx_Oracle
import yaml
from .settings import Settings
import os

settings = Settings()

class Connect:
    def __init__(
        self,
        connection_type=None,
        server=settings.WPH_SERVER,
        database=settings.WPH_DATABASE,
        port=None,
        username=None,
        password=None,
    ):
        # Connection
        if connection_type is None:
            self.server = server
            self.database = database

            print(self.server)

            if port:
                self.port = port
            elif self.server == settings.EDW_SERVER:
                self.port = settings.EDW_PORT
            elif self.server == settings.WPH_SERVER:
                self.port = settings.WPH_PORT

            print(self.port)
        elif connection_type == 'wph_dw':
            self.server = settings.WPH_SERVER
            self.database = settings.WPH_DATABASE
            self.port = settings.WPH_PORT
        elif connection_type == 'mit_edw':
            self.server = settings.EDW_SERVER
            self.database = settings.EDW_DATABASE
            self.port = settings.EDW_PORT

        # Authentication
        self.username = username
        self.password = password

        # Create the connection
        self.conn = self.create_connection()

    def __enter__(self):
        return self

    def close(self):
        self.conn.close()

    def set_connection_string(self):
        if self.server == settings.WPH_SERVER:
            driver = '{ODBC Driver 17 for SQL Server}'

            self.connection_string = (
                f'Driver={driver};'
                f'Server={self.server};'
                f'Database={self.database};'
                f'Trusted_Connection=yes;'
                'MARS_Connection=yes;'
            )
        elif self.server == settings.EDW_SERVER:
            self.connection_string = (
                f'oracle+cx_oracle:'
                f'//{self.username}:{self.password}'
                f'@{self.server}:{self.port}/'
                f'?service_name={self.database}'
            )

    def create_connection(self):
        self.set_connection_string()

        try:
            if self.server == settings.WPH_SERVER:
                connection = pyodbc.connect(self.connection_string)
            elif self.server == settings.EDW_SERVER:

                dll_dir = os.path.join(os.path.dirname(__file__), 'oracle_dlls')

                try:
                    cx_Oracle.init_oracle_client(lib_dir=dll_dir)
                except cx_Oracle.ProgrammingError as err:
                    error, = err.args

                    if 'already been initialized' in error.message:
                        pass
                    else:
                        raise err

                engine = sa.create_engine(self.connection_string)
                connection = engine.connect()
        except pyodbc.Error as err:
            print(f'Could not connect!: {err}')
        except Exception as err:
            print(f'Could not connect!: {err}')

        return connection

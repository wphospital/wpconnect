import pyodbc
import sqlalchemy as sa
import cx_Oracle
from .settings import Settings
import os
import platform
from urllib.parse import quote_plus

settings = Settings()

class Connect:
    def __init__(
        self,
        connection_type=None,
        environ='dev',
        server=settings.WPH_SERVER,
        database=settings.WPH_DATABASE,
        port=None,
        username=None,
        password=None,
        trusted_connection=True,
    ):
        self.trusted_connection = trusted_connection

        # Connection
        if connection_type is None:
            self.server = server
            self.database = database

            if port:
                self.port = port
            elif self.server in [settings.EDW_SERVER, settings.EDW_SERVER_PROD]:
                self.port = settings.EDW_PORT
            elif self.server == settings.WPH_SERVER:
                self.port = settings.WPH_PORT
            else:
                self.port = 1433
        elif connection_type == 'wph_dw':
            self.server = settings.WPH_SERVER
            self.database = settings.WPH_DATABASE
            self.port = settings.WPH_PORT
        elif connection_type == 'wph_tt':
            self.server = settings.TT_SERVER
            self.database = settings.TT_DATABASE
            self.port = settings.TT_PORT
        elif connection_type == 'mit_edw':
            self.port = settings.EDW_PORT

            if environ == 'dev':
                self.server = settings.EDW_SERVER
                self.database = settings.EDW_DATABASE_DEV
            elif environ == 'qa':
                self.server = settings.EDW_SERVER
                self.database = settings.EDW_DATABASE_QA
            else:
                self.server = settings.EDW_SERVER_PROD
                self.database = settings.EDW_DATABASE_PROD

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
        if self.password:
            safe_password = quote_plus(self.password)

        if self.server in [settings.EDW_SERVER, settings.EDW_SERVER_PROD]:
            self.connection_string = (
                f'oracle+cx_oracle:'
                f'//{self.username}:{safe_password}'
                f'@{self.server}:{self.port}/'
                f'?service_name={self.database}'
            )
        else:
            driver = '{ODBC Driver 17 for SQL Server}'

            if self.trusted_connection:
                auth_str = 'Trusted_Connection=yes;'
            else:
                auth_str = f'UID={self.username};PWD={safe_password};'

            self.connection_string = (
                f'Driver={driver};'
                f'Server={self.server};'
                f'Database={self.database};'
                f'{auth_str}'
                'MARS_Connection=yes;'
            )

    def create_connection(self):
        self.set_connection_string()

        try:
            if self.server in [settings.EDW_SERVER, settings.EDW_SERVER_PROD]:

                dll_dir = os.path.join(os.path.dirname(__file__), 'oracle_dlls')

                if platform.system() == 'Windows':
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
            else:
                connection = pyodbc.connect(self.connection_string)
        except pyodbc.Error as err:
            print(f'Could not connect!: {err}')
        except Exception as err:
            print(f'Could not connect!: {err}')

        return connection

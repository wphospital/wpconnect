class Settings:
    # WPH database configs
    WPH_SERVER = '10.255.33.34'
    WPH_DATABASE = 'WPH_DW'
    WPH_SCHEMA = 'dbo'
    WPH_PORT = 1433

    # Teletracking database configs
    TT_SERVER = 'SHNPTCWTLDPVM01\SQL2016PROD'
    TT_DATABASE = 'dNa'
    TT_PORT = 1433

    # MIT EDW configs
    EDW_SERVER = 'od01-scan.montefiore.org'
    EDW_SERVER_PROD = 'yk02-scan.montefiore.org'
    EDW_DATABASE_DEV = 'EDWDEV.montefiore.org'
    EDW_DATABASE_QA = 'EDWADG.montefiore.org'
    EDW_DATABASE_PROD = 'EDWPRD.montefiore.org'
    EDW_PORT = 3923

    # IO configs
    DEFAULT_RETURN = 'DataFrame'

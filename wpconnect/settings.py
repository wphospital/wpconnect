from sprucepy import secrets

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
    EDW_SERVER_DEV = 'od02-scan.montefiore.org'
    EDW_SERVER_QA = 'od02-scan.montefiore.org'
    EDW_SERVER_PROD = 'yk02-scan.montefiore.org'
    EDW_DATABASE_DEV = 'EDWDEV.montefiore.org'
    EDW_DATABASE_QA = 'EDWDG.montefiore.org'
    EDW_DATABASE_PROD = 'EDWPRD.montefiore.org'
    EDW_PORT = 3923

    # IO configs
    DEFAULT_RETURN = 'DataFrame'

    # WPAPI Redis configs
    WPAPI = 'http://odcwphvlp03:1597/api/'
    WPAPI_REDIS_HOST = 'odcwphvlp03'
    WPAPI_REDIS_PORT = 6380
    WPAPI_REDIS_PASSWORD = secrets.get_secret_by_key(
        'wpconnect_redis_password',
        api_url='http://ODCWPHVLD01:1592/api/v1/'
    )

    GITHUB_RATE_WARNING_THRESHOLD = 0.1
    GITHUB_RATE_PAUSE = 1
    GITHUB_RATE_KILL = 2

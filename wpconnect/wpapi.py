from cachelib.redis import RedisCache
import requests
from requests.auth import HTTPBasicAuth
import pickle
import warnings

import pandas as pd

from .settings import Settings

settings = Settings()

def get_precache_list():
    """Get the current list of precache configs from WPAPI

    Returns
    -------
    pandas.DataFrame
        A dataframe listing the current precache configs
    """

    res = requests.get(settings.WPAPI + 'precache_list')

    if res.status_code == 200:
        return pd.DataFrame(res.json())
    else:
        return res.text


class WPAPIResponse:
    def __init__(self, **kwargs):
        for _, k in kwargs.items():
            self.__dict__[_] =  k

        self.iserror = False

    def set_query(self, query):
        self._query = query

    def set_data(self, data, key):
        self._data = data
        self._key = key

    def get_data(self):
        if self.iserror:
            return self._error
        else:
            if isinstance(self._data, list):
                df = pd.concat([pickle.loads(d) for d in self._data])
            else:
                df = pickle.loads(self._data)

        df.__cached__ = self.cached

        return df

    def set_error(self, error):
        self.iserror = True

        self._error = error

class WPAPIRequest:
    def __init__(self, password, prefix='flask_cache_', endpoint='repo_query'):
        self.cache = self.init_redis(password)

        self.prefix = prefix

        self.endpoint = endpoint

    def init_redis(self, password):
        return RedisCache(
            host=settings.WPAPI_REDIS_HOST,
            port=settings.WPAPI_REDIS_PORT,
            password=password
        )

    @staticmethod
    def package_params(params):
        return ','.join(['{}={}'.format(
            k,
            '({})'.format(
                ';'.join([f'\'{i}\'' for i in v])
            ) if isinstance(v, list) else v
        ) for k, v in params.items()])

    def get(
        self,
        query_fn : str = None,
        query_params : dict = None,
        headers : dict = None,
        auth : tuple = None,
        **kwargs
    ):
        self.last_query_fn = query_fn
        self.last_params = kwargs

        default_dict = {'query_fn': query_fn} if query_fn else {}

        send_params = {
            **default_dict,
            **{
                'return_cache_key': True,
                'return_query': True
            },
            **kwargs
        }

        if query_params:
            send_params = {
                **send_params,
                **{'query_params': self.package_params(query_params)}
            }

        basic = None if auth is None else HTTPBasicAuth(*auth)

        res = requests.get(
            settings.WPAPI + self.endpoint,
            params=send_params,
            headers=headers,
            auth=basic
        )

        resp = WPAPIResponse(
            cached=res.headers.get('data-cached') == 'True',
            status_code=res.status_code,
            request_res=res
        )

        try:
            self.last_query = res.json()['query']
        except Exception as err:
            self.last_query = None

        resp.set_query(query=self.last_query)

        if res.status_code == 200:
            self.last_key = res.json()['data']

            if len(self.last_key) == 1:
                data = self.cache.get(self.prefix + self.last_key[0])
            else:
                data = [self.cache.get(self.prefix + k) for k in self.last_key]

            resp.set_data(data=data, key=self.last_key)
        else:
            try:
                warnings.warn(res.json()['data'])
                resp.set_error(res.json()['data'])
            except Exception as err:
                warnings.warn(res.text)
                resp.set_error(res.text)

        return resp

from cachelib.redis import RedisCache
import requests
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3 import Retry
import pickle
import zlib
import warnings

import pandas as pd

from .settings import Settings

import gc

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

def retry_request(
    url,
    params,
    headers,
    auth : tuple = None,
    retries : int = 3,
    backoff_factor : float = 0.5,
    status_forcelist : tuple = (500, 502, 503, 504)
):
    """
    Sends an HTTP GET request to a URL with retry logic.

    Args:
        url (str): The URL to request.
        retries (int, optional): Maximum number of retry attempts. Defaults to 3.
        backoff_factor (float, optional): Backoff factor for delay between retries. Defaults to 0.5.
        status_forcelist (tuple, optional): HTTP status codes to retry on. Defaults to (500, 502, 503, 504).

    Returns:
        requests.Response: The response object if successful or after max retries.
    """
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    response = http.get(url, params=params, headers=headers, auth=auth)

    try:
        if response.status_code not in (401, 404):
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        
        return response
    except requests.exceptions.RequestException as e:
        base_err_str = f'WPConnect Error: WPConnect did not receive a successful response after {retries} attempts'

        try:
            err_str = base_err_str + '. ' + response.text
        except NameError as err:
            err_str = base_err_str

        try:
            return_val = response
        except NameError as err:
            return_val = requests.Response()

            return_val.status_code = 444

            return_val._content = err_str.encode()

        return return_val


class WPAPIResponse:
    def __init__(self, **kwargs):
        for _, k in kwargs.items():
            self.__dict__[_] =  k

        self.iserror = False

    def set_query(self, query):
        self._query = query

    def set_data(self, data, key, from_cache : bool = True):
        self._data = data
        self._key = key
        self._from_cache = from_cache

    @staticmethod
    def decompress(obj):
        try:
            obj = zlib.decompress(obj)
        except TypeError as err:
            pass
        except zlib.error as err:
            pass

        return obj

    def get_data(
        self,
        as_data_frame : bool = True
    ):
        if self.iserror:
            return self._error
        else:
            if self._from_cache:
                if isinstance(self._data, list):
                    df_list = []
                    for d in self._data:
                        dcmp = self.decompress(d)

                        df_part = pickle.loads(dcmp)

                        df_list.append(df_part)

                        del dcmp
                        del df_part

                    df = pd.concat(df_list)

                    del df_list
                else:
                    dcmp = self.decompress(self._data)

                    df = pickle.loads(dcmp)

                    del dcmp
            else:
                if as_data_frame:
                    df = pd.DataFrame(self._data)
                else:
                    df = self._data

        try:
            df._attrs['cached'] = self.cached
        except Exception as err:
            pass

        gc.collect()

        return df

    def set_error(self, error):
        self.iserror = True

        self._error = error

class WPAPIRequest:
    def __init__(
        self,
        password=None,
        prefix='flask_cache_',
        endpoint='repo_query',
        auth=None
    ):
        if password is not None:
            warnings.warn(
                'Passing a password to WPAPIRequest is depracated. '
                'Please switch to pass basic auth tuple to WPAPIRequest',
                DeprecationWarning,
                stacklevel=2
            )

        self.cache = self.init_redis(settings.WPAPI_REDIS_PASSWORD)

        self.prefix = prefix

        self.endpoint = endpoint

        self.auth = auth

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

        if auth is not None:
            warnings.warn(
                'Passing basic auth to the get method is depracated. '
                'Please switch to pass basic auth tuple to WPAPIRequest',
                DeprecationWarning,
                stacklevel=2
            )
        else:
            auth = self.auth

        default_dict = {'query_fn': query_fn}\
            if query_fn and self.endpoint != 'named_query'\
            else {}

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

        res = retry_request(
            url=settings.WPAPI + self.endpoint + (
                f'/{query_fn}' if self.endpoint == 'named_query' else ''
            ),
            params=send_params,
            headers=headers,
            auth=basic
        )

        res_headers = res.headers if res.headers is not None else {}

        resp = WPAPIResponse(
            cached=res_headers.get('data-cached') == 'True',
            status_code=res.status_code,
            request_res=res
        )

        try:
            self.last_query = res.json()['query']
        except Exception as err:
            self.last_query = None

        resp.set_query(query=self.last_query)

        if res.status_code == 200:
            if kwargs.get('return_cache_key', True):
                self.last_key = res.json()['data']

                if len(self.last_key) == 1:
                    data = self.cache.get(self.prefix + self.last_key[0])
                else:
                    data = []
                    for k in self.last_key:
                        get_part = self.cache.get(self.prefix + k)

                        data.append(get_part)

                resp.set_data(data=data, key=self.last_key)

                del data
            else:
                res_json = res.json()

                if isinstance(res_json, dict):
                    data = res_json.get('data', res_json)
                else:
                    data = res_json

                resp.set_data(data=data, key=None, from_cache=False)
        else:
            try:
                warnings.warn(res.json()['data'])
                resp.set_error(res.json()['data'])
            except Exception as err:
                warnings.warn(res.text)
                resp.set_error(res.text)

        gc.collect()

        return resp

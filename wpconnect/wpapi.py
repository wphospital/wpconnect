from cachelib.redis import RedisCache
import requests
import pickle

from .settings import Settings

settings = Settings()

class WPAPIResponse:
    def __init__(self, **kwargs):
        for _, k in kwargs.items():
            self.__dict__[_] =  k

        self.iserror = False

    def set_data(self, data, key):
        self._data = data
        self._key = key

    def get_data(self):
        if self.iserror:
            return self._error
        else:
            return pickle.loads(self._data)

    def set_error(self, error):
        self.iserror = True

        self._error = error

class WPAPIRequest:
    def __init__(self, password, prefix='flask_cache_'):
        self.cache = self.init_redis(password)

        self.prefix = prefix

    def init_redis(self, password):
        return RedisCache(
            host=settings.WPAPI_REDIS_HOST,
            port=settings.WPAPI_REDIS_PORT,
            password=password
        )

    def get(self, query_fn, **kwargs):
        self.last_query_fn = query_fn
        self.last_params = kwargs

        res = requests.get(
            settings.WPAPI + 'repo_query',
            params={
                **{
                    'query_fn': query_fn,
                    'return_cache_key': True
                },
                **kwargs
            }
        )

        res = WPAPIResponse(
            cached=res.headers.get('data-cached'),
            status_code=res.status_code
        )

        if res.status_code == 200:
            self.last_key = res.json()

            data = self.cache.get(self.prefix + self.last_key)

            res.set_data(data=data, key=self.last_key)
        else:
            res.set_error(res.text)

        return res

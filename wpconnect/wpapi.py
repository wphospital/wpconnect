from cachelib.redis import RedisCache
import requests
import pickle

from .settings import Settings

settings = Settings()

class WPAPIResponse:
    def __init__(self, **kwargs):
        for _, k in kwargs.items():
            self.__dict__[_] =  k

    def set_data(self, data):
        self._data = data

    def get_data(self):
        return pickle.loads(self._data)

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

        key = res.json()

        self.last_key = key

        data = self.cache.get(self.prefix + key)

        res = WPAPIResponse(
            key=key,
            cached=res.headers.get('data-cached'),
            status_code=res.status_code
        )

        res.set_data(data=data)

        return res

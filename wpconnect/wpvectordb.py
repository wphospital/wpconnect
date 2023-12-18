import pandas as pd

from pandas.api.types import is_datetime64_any_dtype as is_datetime

import warnings
import weaviate
from weaviate.util import generate_uuid5
from .wpgraph import GraphResult

import time

def configure_batch(client: weaviate.Client, batch_size: int, batch_target_rate: int):
    """
    Configure the weaviate client's batch so it creates objects at `batch_target_rate`.

    Parameters
    ----------
    client : Client
        The Weaviate client instance.
    batch_size : int
        The batch size.
    batch_target_rate : int
        The batch target rate as # of objects per second.
    """

    def callback(batch_results: dict) -> None:

        weaviate.util.check_batch_result(batch_results)

        time_took_to_create_batch = batch_size * (client.batch.creation_time/client.batch.recommended_num_objects)
        
        time.sleep(
            max(batch_size/batch_target_rate - time_took_to_create_batch + 1, 0)
        )

    client.batch.configure(
        batch_size=batch_size,
        timeout_retries=5,
        callback=callback,
    )

class VectorDB:
    """This module is facilitate interactions with weaviate database, 
        mainly Provider, Patients and Visits embeddings.

    Attributes
    ----------
    client  
    """
    
    def __init__(
        self, 
        url,
        auth_key,
        **kwargs
    ):
        self.client = weaviate.Client(
            url, 
            auth_client_secret=weaviate.AuthApiKey(api_key=auth_key), 
            **kwargs
        )

        configure_batch(self.client, batch_size=5, batch_target_rate=1)
        
    def _get_key(self, kwargs : dict):
        """Helper function to generate the uuid used in row identifier inside class
        
        Returns: uuid
        """
        return generate_uuid5(kwargs)
    
    def get_rowcount(self, class_name=None):
        """List number of records in class

        Parameters
        ----------
        class_name: table name. e.g.: Patients, Visits, Providers
        
        Returns
        -------
        if class_name is None, returns a dictionary with key being the class name and value being the row counts;
        otherwise returns a rowcount number 

        """
        def get_n(class_name):
            n = self.client.query\
            .aggregate(class_name)\
            .with_meta_count()\
            .do()["data"]["Aggregate"][class_name][0]["meta"]["count"]
            return n
        
        if class_name:
            class_name = class_name.capitalize()
            return get_n(class_name)
            
        res = {}
        classes = [x['class'] for x in self.client.schema.get(class_name)['classes']]
        for c in classes:
            res[c] = get_n(c)
        return res


    def get_schema(self, class_name=None):
        """Shows class schema 

        Parameters
        ----------
        class_name: table name. e.g.: Patients, Visits, Providers
        
        Returns
        -------
        if class_name is None, returns a dictionary of dataframes with key being class name in the database, 
            dataframe contains column name and type of that class
        otherwise returns a dataframe of column name and type of the given class
        """
        
        def _get_schema_by_class(res):
            df = pd.DataFrame([
                {'column_name':x['name'],'data_type':x['dataType']} 
                for x in res['properties']
                ])
            return df 

        if  class_name:
            class_name = class_name.capitalize()
            res = [self.client.schema.get(class_name)]
        else:
            res = self.client.schema.get()['classes']
            
        final_dict = {}
        
        for i in range(len(res)):
            class0 = res[i]
            cn = class0['class']
            final_dict[cn] = {}
            final_dict[cn]  = _get_schema_by_class(class0)
            # final_dict[cn]['nrows'] = self.get_rowcount(cn)
        if class_name:
            return final_dict[class_name]
        return final_dict

    def delete_class(self, class_name):
        """Delete the given class in database

        Parameters
        ----------
        class_name: the class to be deleted
        """
        class_name = class_name.capitalize()
        self.client.schema.delete_class(class_name)

    def create_class(self, class_name,**kwargs):
        """Create a new class in the database

        Parameters
        ----------
        class_name: the class to be created
        """
        class_name = class_name.capitalize()
        obj = {'class':class_name}
        obj.update(kwargs)
        self.client.schema.create_class(obj)

    @staticmethod
    def _check_datetime(value):
        try:
            return is_datetime(value)
        except ValueError as err:
            return False

    def process_val(self, value):
        if self._check_datetime(value) or isinstance(value, pd.Timestamp) or value is pd.NaT:
            return value.isoformat()

        return value

    def load_data(
        self,
        dat,
        class_name:str,
        source: str = None,
        has_admissions: int = 1,
        uuid_keys : list = ['db_id'],
        custom_embeddings: str = 'embedding',
        **kwargs
    ):
        """Import data to a class

        Parameters
        ----------
        dat: List or DataFrame or GraphResult 
            data source to be imported
        class_name: Str 
            the import destination class name
        source: str
            type of the record. e.g. :"provider", "patient", "inp", "ed", etc.
        has_admissions: int
            0 or 1 indicating if this embedding contain admission info 
        """
        if 'source' not in dat.columns and source is None:
            raise Exception('Must provide a source either in the data or as an argument')

        if source is not None:
            if 'source' in dat.columns:
                warnings.warn('Source provided as an argument. Overriding values in data')

        class_name = class_name.capitalize()
        if not self.client.schema.exists(class_name):
            print("create class")
            self.create_class(class_name, **kwargs)

        class_cols = self.get_columns(class_name)

        load_cols = set(class_cols).union(set(dat.columns))

        if isinstance(dat, pd.DataFrame):
            dat = dat.to_dict('records')
        elif isinstance(dat, GraphResult):
            dat = dat.records

        with self.client.batch as batch:
            for _, r in enumerate(dat):
                row_source = source if source is not None else r['source']

                all_keys = {
                    **{'source': row_source},
                    **{u: r.get(u) for u in uuid_keys},
                    **{'has_admissions': has_admissions}
                }

                if custom_embeddings:

                    obj = {
                        **{'source': row_source, 'has_admissions': has_admissions},
                        **{
                            k: self.process_val(r.get(k))
                            for k in load_cols if k != custom_embeddings
                        }
                    }

                    batch.add_data_object(
                        obj,
                        class_name,
                        vector=r.get("embedding"),
                        uuid=self._get_key(all_keys)
                    )
                else:
                    obj = {
                        **{'source': row_source, 'has_admissions': has_admissions},
                        **{
                            k: self.process_val(r.get(k))
                            for k in load_cols
                        }
                    }

                    batch.add_data_object(
                        obj,
                        class_name,
                        uuid=self._get_key(all_keys)
                    )

                if _ % 100 == 0:
                    print(_)

    def get_records(self, 
            class_name: str, 
            db_id: int=None, 
            native_id: str=None, 
            has_admissions: int=1, 
            source: str=None, 
            columns: list=None,
            uuid_keys : list = ['db_id']
        ):
        """To retrive embeddings from Weaviate database. 

        Parameters
        ----------
        class_name: str
            which class to retrive the record from. e.g.: Patients, Visits, Providers
        db_id: int
        native_id: str
        has_admissions: int
            0: with no admissions, 1:with admissions. 1 being default
        source: str
            type of record. for Patients, source="patient", 
                            for Visits, source is either "inp" or "ed", 
                            for Providers, source="provider". 
            if not given, it's inferred from class_name. will be "inp" for Visits by default
        columns: list of strings
            additional columns to return besides embedding and id 
            if not given, returns all columns

        Returns
        -------
        Dataframe with embedding, id and columns defined in the parameter
        """
        class_name = class_name.capitalize()
        if not columns:
            columns = self.get_columns(class_name)
        elif isinstance(columns,str):
            columns = [columns]

        if class_name == 'Patients':
            source='patient'
        elif class_name == 'Providers':
            source = 'provider'
        elif class_name == 'Comments':
            source = 'comment'
        elif not source:
            source='inp'
        
        if not db_id and not native_id:
            
            cursor = None
            
            df = pd.DataFrame(columns=columns)
            df['embedding'] = None
            def get_batch_with_cursor(class_name, columns, cursor=None):
                
                query = (
                    self.client.query.get(class_name, columns)
                    .with_additional(["id vector"])
                    .with_limit(500)
                )
            
                if cursor is not None:
                    return query.with_after(cursor).do()
                else:
                    return query.do()
        
            while True:
                results = get_batch_with_cursor(class_name, columns, cursor)
                objects_list= results["data"]["Get"][class_name] 
                if results is None or len(objects_list) == 0:
                    break
                cursor = objects_list[-1]["_additional"]["id"] 
                
                _df = pd.DataFrame([{c:x[c] for c in columns} for x in objects_list])
                _df['embedding'] = [x['_additional']['vector'] for x in objects_list]
                _df['id'] = [x['_additional']['id'] for x in objects_list]
                
                df = pd.concat([df,_df])

            df = df.query('source==@source and has_admissions==@has_admissions')       
            
        elif db_id:
            
            all_keys = {
                'source': source,
                'db_id': db_id,
                'has_admissions': has_admissions
            }

            uuid = self._get_key(all_keys)
            dat = self.client.data_object.get_by_id(uuid,class_name=class_name,with_vector=True )

            try:
                df = pd.DataFrame(dat['properties'],index=[0],columns=columns)
                df['embedding'] = [dat['vector']]
                df['id'] =  dat['id']
            except TypeError:
                print('no record found')
                df = pd.DataFrame()       
            
        else:
            objects_list = (self.client.query.get(class_name,columns)
                   .with_where({
                       "operator": "And",
                       "operands": [
                            {'path':'native_id','operator':'Equal','valueText':native_id},
                           {"operator":"And",
                            "operands":[
                                {'path':'has_admissions','operator':'Equal','valueInt':has_admissions},
                                {'path':'source','operator':'Equal','valueText':source}
                                ]}
                            ]
                   })
                   .with_additional(["id vector"])
                   .do()['data']['Get'][class_name]) 
            
            try:
                df = pd.DataFrame([{c:x[c] for c in columns} for x in objects_list])
                df['embedding'] = [x['_additional']['vector'] for x in objects_list]
                df['id'] =  [x['_additional']['id'] for x in objects_list]
            except TypeError:
                print('no record found')
                df = pd.DataFrame()
                
        return df 

    def get_columns(self, class_name:str):
        """To list columns of the given class 

        Parameters
        ----------
        class_name: str
            which class to retrive the info from. e.g.: Patients, Visits, Providers

        Returns
        -------
        List of column names in the class
        """

        class_name = class_name.capitalize()
        c = [x['name'] for x in self.client.schema.get(class_name)['properties']]
        return c

    def get_distinct_values(self, class_name: str, column_name:str):
        """To list unique values of the given column 

        Parameters
        ----------
        class_name: str
            which class to retrive the info from. e.g.: Patients, Visits, Providers
        column_name: str
            which column to search for distinct values

        Returns
        -------
        List of distinct values in the column
        """
        class_name = class_name.capitalize()
        vals = (
            self.client.query
            .aggregate(class_name)
            .with_group_by_filter([column_name])
            .with_fields("groupedBy { value }")
            .do()
        )['data']['Aggregate'][class_name]

        return [i['groupedBy']['value'] for i in vals]

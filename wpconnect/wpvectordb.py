import pandas as pd 
import weaviate
from weaviate.util import generate_uuid5
from .wpgraph import GraphResult

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
        
    def _get_key(self, source, db_id, has_admissions=1):
        """Helper function to generate the uuid used in row identifier inside class
        
        Returns: uuid
        """
        return generate_uuid5({'source':source, 'db_id':db_id, 'has_admissions':has_admissions})
    
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

    def load_data(self, dat, class_name:str, source: str, has_admissions: int=1, **kwargs):
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
        class_name = class_name.capitalize()
        if not self.client.schema.exists(class_name):
            print("create class")
            self.create_class(class_name, **kwargs)

        if isinstance(dat, pd.DataFrame):
            dat = dat.to_dict('records')
        elif isinstance(dat, GraphResult):
            dat = dat.records

        with self.client.batch as batch:
            for r in dat:
                if class_name == "Visits":
                    obj = {
                        "db_id":r["db_id"],
                        "native_id":r["native_id"],
                        "start_date":r["start_date"].isoformat() if r["start_date"] else None,
                        "end_date":r["end_date"].isoformat() if r["end_date"] else None,
                        "source":source,
                        "has_admissions":has_admissions}
                elif class_name == "Patients":
                    source = "patient"
                    obj = {
                        "native_id":r["native_id"],
                        "db_id":r["db_id"],
                        "has_admissions": has_admissions,
                        "source":source
                    }

                elif class_name == "Providers":
                    source = "provider"
                    obj = {
                        "native_id":r["native_id"],
                        "db_id":r["db_id"],
                        "has_admissions": has_admissions,
                        "source":source
                    }
                else:
                    obj = {k:v for k,v in r.items() if k != 'embedding'}
                    obj.update({'has_admissions':has_admissions, "source":source})


                batch.add_data_object(
                    obj,
                    class_name,
                    vector=r["embedding"],
                    uuid=self._get_key(source, r["db_id"], has_admissions)
                )
        # print(self.client.query\
        #     .aggregate(class_name)\
        #     .with_where({
        #         "operator":"And",
        #         "operands":[
        #             {'path':'has_admissions','operator':'Equal','valueInt':has_admissions},
        #             {'path':'source','operator':'Equal','valueText':source}
        #             ]
        #         })\
        #     .with_meta_count()\
        #     .do() )

    def get_records(self, 
            class_name: str, 
            db_id: int=None, 
            native_id: str=None, 
            has_admissions: int=1, 
            source: str=None, 
            columns: list=None
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
            columns = [x['name'] for x in self.client.schema.get(class_name)['properties']]
        elif isinstance(columns,str):
            columns = [columns]

        if class_name == 'Patients':
            source='patient'
        elif class_name == 'Providers':
            source = 'provider'
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
            
            uuid = self._get_key(source, db_id, has_admissions)
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

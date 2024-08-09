# VectorDB/Weaviate database
- Comments data, patient embeddings, provider embeddings and visit embeddings are all stored here. To get the schema, run the following: 
  ~~~
  v = VectorDB(
    cfg['vectordb'],
    sd['weaviate_key'],
    **{'additional_headers':{"X-HuggingFace-Api-Key": sd['hf_api_token']}}
  )
  v.get_schema() ## this shows a concise view of all tables, columns within each table, and the data types

  v.client.schema.get() ## this includes all detailed configurations
  ~~~
- For more information, please click [here](https://weaviate.io/developers/weaviate/tutorials) for official documentation and [here](https://github.com/wphospital/wpconnect/blob/main/wpconnect/wpvectordb.py) for additional helper methods.  
 

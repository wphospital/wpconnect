from neo4j import GraphDatabase

import pandas as pd
import numpy as np

from pandas.api.types import is_any_real_numeric_dtype as is_numeric, is_object_dtype

from itertools import chain
from collections import Counter

from datetime import timedelta
import time
import pytz

import logging

import re

# Create a logger
logger = logging.getLogger(__name__)

# Set the level of the logger
logger.setLevel(logging.INFO)

# Create a handler to write to stdout
handler = logging.StreamHandler()

# Set the format of the handler
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add the handler to the logger
logger.addHandler(handler)

class GraphResult:
    """
    A data wrapper class holding graph records and conversion methods

    Attributes
    ----------
    records : list
        the list of graph Records returned from the database

    Methods
    -------
    to_pandas()
        Converts the graph records to a dataframe
    """

    def __init__(self, records : list = []):
        self.records = records

    @staticmethod
    def _cleanup_keys(df : pd.DataFrame):
        df = df\
            .astype({
                k: int
                for k in df.columns
                if is_numeric(df[k]) and ('id' in k or 'key' in k)
            })

        return df

    @staticmethod
    def _convert_dates(df : pd.DataFrame):
        for c in [c for c in df.columns if is_object_dtype(df[c])]:
            try:
                df[c] = pd.to_datetime(df[c], format='ISO8601')
            except ValueError as err:
                pass
        
        for c in [c for c in df.columns if isinstance(df[c].values[0], pd.Timestamp)]:
            df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert(pytz.timezone('America/New_York'))
            
        return df

    def _df_cleanup(self, df : pd.DataFrame):
        df = self._cleanup_keys(df)

        df = self._convert_dates(df)

        return df

    def to_pandas(
        self,
        recursive : bool = True,
        index_cols : list = None
    ):
        """Converts the graph records to a dataframe

        Parameters
        ----------
        recursive : bool
            whether nested keys should be pulled out
        """

        if len(self.records) == 0:
            return pd.DataFrame()

        infer_item = self.records[0]

        record_keys = infer_item.keys()

        keys_with_keys = {}

        for k in record_keys:
            try:
                keys_with_keys[k] = list(infer_item[k].keys())
            except AttributeError as err:
                pass

        if recursive and len(keys_with_keys) > 0:
            df_list = []
            for _, r in enumerate(self.records):
                rec_fr = pd.DataFrame()
                for k in record_keys:
                    if k in keys_with_keys.keys():
                        for ik in r[k].keys():
                            val = r[k].get(ik)
                            
                            basename = (str(k) + '.' if len(keys_with_keys) > 1 else '') + str(ik)

                            if isinstance(val, list):
                                colnames =  [basename + '_' + str(_) for _ in range(0, len(val))]

                                subdf = pd.DataFrame({c: val[_] for _, c in enumerate(colnames)}, index=[_])
                            else:
                                subdf = pd.DataFrame.from_records({basename: val}, index=[_])
                        
                            rec_fr = pd.concat([rec_fr, subdf], axis=1)
                    else:
                        val = r.get(k)

                        if isinstance(val, list):
                            colnames = [k + '_' + str(_) for _ in range(0, len(val))]

                            subdf = pd.DataFrame({c: val[_] for _, c in enumerate(colnames)}, index=[_])
                        else:
                            subdf = pd.DataFrame({k: r[k]}, index=[_])

                        rec_fr = pd.concat([rec_fr, subdf], axis=1)
                    
                df_list.append(rec_fr)

            rec_fr = pd.concat(df_list)
        else:
            rec_fr = pd.DataFrame({
                k: [r.get(k) for r in self.records]
                for k in self.records[0].keys()
            })

        rec_fr = self._df_cleanup(rec_fr)

        if index_cols is not None:
            rec_fr = rec_fr.set_index(index_cols)
        
        return rec_fr

class GraphSchema:
    schema_set = False

    def __init__(
        self,
        clinical_graph,
        initialize_schema : bool = False,
        verbose : bool = True
    ):
        self.clinical_graph = clinical_graph

        self.verbose = verbose

        if initialize_schema:
            self.set_schema()

    def _set_node_counts(self):
        node_cypher = '''
            //Node Counts
            MATCH (n)
            UNWIND LABELS(n) AS node_label
            RETURN node_label, COUNT(n) AS node_count
            ORDER BY node_count DESC
        '''

        self.nodes = self.clinical_graph.run_cypher(node_cypher).to_pandas(index_cols='node_label')

    def _set_node_props(self):
        prop_cypher = '''
            //Node Props
            MATCH (n)
            UNWIND keys(n) AS node_key
            WITH DISTINCT labels(n) AS node_labels, node_key
            ORDER BY node_labels, node_key
            WITH node_labels, COLLECT(node_key) AS node_props
            UNWIND node_labels AS node_label
            RETURN node_label, node_props
        '''

        self.node_props = self.clinical_graph.run_cypher(prop_cypher).to_pandas(index_cols='node_label')


    def set_schema(
        self,
        nodes : list = None,
        relationships : list = None
    ):
        if self.verbose:
            logger.info('Scanning graph schema')

            start = time.time()

        self._set_node_counts()

        self._set_node_props()

        if self.verbose:
            elapsed_time = time.time() - start

            elapsed = str(timedelta(seconds=elapsed_time))

            logger.info(f'Schema scan completed in {elapsed}')

        self.schema_set = True


class ClinicalGraph:
    def __init__(
        self,
        uri : str,
        user : str,
        password : str,
        initialize_schema : bool = False
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        self.schema = GraphSchema(self, initialize_schema=initialize_schema)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def close(self):
        self.driver.close()

    @staticmethod
    def _run_cypher(tx, cypher_string):
        result = tx.run(cypher_string)
        
        return list(result)

    def run_cypher(self, cypher_string):
        with self.driver.session() as session:
            result = session.execute_write(self._run_cypher, cypher_string)
            
        return GraphResult(result)

    def get_all_nodes(
        self,
        tags : str = None
    ):
        if tags:
            tag_str = ':' + ('|'.join(tags) if isinstance(tags, list) else tags)
        else:
            tag_str = ''

        return self.run_cypher(f'MATCH (n{tag_str}) RETURN n')

    @staticmethod
    def _match_pattern_exact(str_val, pat):
        if isinstance(str_val, list):
            return pat in str_val, pat
        elif isinstance(str_val, str):
            return pat == str_val, pat
        else:
            return False

    @staticmethod
    def _match_pattern_regex(str_val, pat, flags=0):
        pat = re.compile(pat, flags) # TODO: handle malformed pattern errors

        if isinstance(str_val, list):
            match_res = [re.search(pat, s) for s in str_val]
            matches = [res.group(0) for res in match_res if res is not None]

            return any([b is not None for b in match_res]), matches
        elif isinstance(str_val, str):
            return re.search(pat, str_val) is not None, re.findall(pat, str_val)
        else:
            return False

    def get_nodes_with_prop(
        self,
        prop : str,
        regex : bool = False
    ):
        if not self.schema.schema_set:
            self.schema.set_schema()

        match_func = self._match_pattern_regex if regex else self._match_pattern_exact

        matches = self.schema.node_props['node_props'].apply(match_func, args=(prop,))

        match_inds = [r[0] for r in matches]
        match_vals = [r[1] for r in matches]

        if len(match_vals) > 0 and isinstance(match_vals[0], list):
            match_vals = list(chain(*match_vals))

        return self.schema.node_props[match_inds], list(set(match_vals))

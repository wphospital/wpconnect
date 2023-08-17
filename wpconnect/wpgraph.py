from neo4j import GraphDatabase

import pandas as pd
import numpy as np

from pandas.api.types import is_any_real_numeric_dtype as is_numeric, is_object_dtype, is_datetime64_any_dtype as is_datetime

from itertools import chain
from collections import Counter

from datetime import timedelta
import time
import pytz

import logging

import warnings

import re
import json

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

    def __init__(
        self,
        records : list = [],
        cypher_string : str = None
    ):
        self.records = records

        self.query = cypher_string

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

    def _set_edge_counts(self):
        edge_cypher = '''
            //Edge Counts
            //Edge Counts
            MATCH (n1) -[e]-> (n2)
            WITH LABELS(n1) AS node1_labels, LABELS(n2) AS node2_labels, TYPE(e) AS edge_type, COUNT(e) AS edge_count
            UNWIND node1_labels AS node1_label
            UNWIND node2_labels AS node2_label
            RETURN node1_label, node2_label, edge_type, edge_count
            ORDER BY edge_count DESC
        '''

        self.edges = self.clinical_graph.run_cypher(edge_cypher).to_pandas()

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

        self._set_edge_counts()

        if self.verbose:
            elapsed_time = time.time() - start

            elapsed = str(timedelta(seconds=elapsed_time))

            logger.info(f'Schema scan completed in {elapsed}')

        self.schema_set = True

    def get_nodes_with_prop(
        self,
        prop : str,
        regex : bool = False
    ):
        if not self.schema_set:
            self.set_schema()

        match_func = self._match_pattern_regex if regex else self._match_pattern_exact

        matches = self.node_props['node_props'].apply(match_func, args=(prop,))

        match_inds = [r[0] for r in matches]
        match_vals = [r[1] for r in matches]

        if len(match_vals) > 0 and isinstance(match_vals[0], list):
            match_vals = list(chain(*match_vals))

        return self.node_props[match_inds], list(set(match_vals))

    def get_edges_from_nodes(
        self,
        node_labels : list
    ):
        if not self.schema_set:
            self.set_schema()

        nodes = self.nodes.query('node_label.isin(@node_labels)')

        return nodes\
            .merge(
                self.edges,
                how='left',
                left_on=['node_label'],
                right_on=['node1_label']
            )


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
    def normalize_value(inputs):
        val = inputs[-1]

        if isinstance(val, str) and 'datetime(' not in val:
            inputs = list(inputs)

            inputs[-1] = f'"{val}"' if not re.search('^(?<=").+(?=")$', val) else val

            return tuple(inputs)
        
        return inputs

    @staticmethod
    def _run_cypher(tx, cypher_string):
        result = tx.run(cypher_string)
        
        return list(result)

    def run_cypher(self, cypher_string):
        with self.driver.session() as session:
            result = session.execute_write(self._run_cypher, cypher_string)
            
        return GraphResult(result, cypher_string)

    # Node retrieval methods

    def _format_node_query(
        self,
        node_tag : str = None,
        return_attrs : dict = 'db_id',
        prop_filter : dict = None,
        filter_list : list = None,
        node_num : int = 0
    ):
        if node_tag:
            node_tag = ':' + ('|'.join(node_tag) if isinstance(node_tag, list) else node_tag)
        else:
            node_tag = ''

        if prop_filter is not None:
            prop = list(prop_filter.keys())[0]
            prop_value = prop_filter[prop]
            
            if len(prop_filter.keys()) > 1:
                warnings.warn(f'Cannot use more than one property filter. Using "{prop}" only.')

            prop_filter_str = ' {' + f'{prop}: "{prop_value}"' + '}'
        else:
            prop_filter_str = ''

        return_str = f'p{node_num}'
        if return_attrs is not None:
            if isinstance(return_attrs, str):
                return_str = f'p{node_num}.{return_attrs} as {return_attrs}'
            elif isinstance(return_attrs, dict):
                return_str = ', '.join(
                    [
                        f'p{node_num}.{k} as {v}'.format()
                        for k, v in return_attrs.items()
                    ]
                )
            elif isinstance(return_attrs, (list, tuple)):
                return_str = ', '.join(
                    [
                        'p{}.{} as {}'.format(
                            node_num,
                            *(n if isinstance(n, (list, tuple)) else (n, n))
                        )
                        for n in return_attrs
                    ]
                )

        filter_str = 'WHERE 1=1'
        if filter_list is not None:
            addon_list = []
            for f in filter_list:
                if len(f) == 3:
                    addon_list.append('p{}.{} {} {}'.format(node_num, *self.normalize_value(f)))
                elif len(f) == 2:
                    addon_list.append('p{}.{} {}'.format(node_num, *f))

            filter_addon = ' AND '.join(addon_list)

            filter_str = f'''
                {filter_str}
                AND {filter_addon}
            '''

        if node_tag is not None:
            return f'(p{node_num}{node_tag}{prop_filter_str})', filter_str, return_str
        else:
            return f'(p{node_num}{prop_filter_str})', filter_str, return_str

    def get_nodes(
        self,
        node_tag : list = None,
        node_props : dict = None,
        prop_filter : dict = None,
        filter_list : list = None,
        node_num : int = 0,
        **kwargs
    ):
        node_str, filter_str, select_str = self._format_node_query(
            node_tag,
            node_props,
            prop_filter,
            filter_list,
            node_num
        )

        query_str = f'''
            MATCH {node_str}
            {filter_str}
            RETURN {select_str}
        '''

        result = self.run_cypher(query_str, **kwargs)
            
        return result

    # Edge retrieval methods

    @staticmethod
    def _format_edge_query(
        edge_tag : str = None,
        return_attrs : dict = 'weight',
        edge_num : int = 0
    ):
        if return_attrs:
            if isinstance(return_attrs, list):
                return_attrs = dict(zip(return_attrs, return_attrs))
            elif isinstance(return_attrs, str):
                return_attrs = {return_attrs: return_attrs}

            select_str = ', '.join(f'e{edge_num}.{a} AS {v}' for a, v in return_attrs.items())
        else:
            select_str = f'e{edge_num}'

        if edge_tag is not None:
            return f'[e{edge_num}:{edge_tag}]', select_str
        else:
            return f'[e{edge_num}]', select_str

    def get_edges(
        self,
        edge_tag : str = None,
        edge_props : dict = 'weight',
        node_tags : tuple = (None, None),
        prop_filters: tuple = (None, None),
        node_props : tuple = (None, None),
        node_filter_lists : tuple = (None, None),
        **kwargs
    ):
        return_nodes = any([r != None for r in node_props])

        node_configs = []
        for _, t in enumerate(node_tags):
            node_configs.append(
                self._format_node_query(
                    node_tag=t,
                    return_attrs=node_props[_],
                    prop_filter=prop_filters[_],
                    filter_list=node_filter_lists[_],
                    node_num=_
                )
            )

        node_addendum = ', ' + ', '.join(
            [nc[2] for nc in node_configs]
        ) if return_nodes else ''

        edge_str, edge_return_str = self._format_edge_query(
            edge_tag=edge_tag,
            return_attrs=edge_props
        )

        filters = [
            (
                nc[1].replace('WHERE 1=1', '')
                if _ == 1 else nc[1]
            ) for _, nc in enumerate(node_configs)
        ]

        joined_filters = ' AND '.join(
            [
                re.sub(
                    ' {2,}',
                    ' ',
                    re.sub('\n', '', f)
                ) for f in filters
            ]
        ) if filters[1] != '' else filters[0]

        filter_addendum = re.sub(
            'AND +AND',
            'AND',
            joined_filters
        )

        query_str = '''
            MATCH {node_str1} -{edge_str}-> {node_str2}
            {filter_str}
            RETURN {edge_return_str}{node_return_str}
        '''.format(
            node_str1=node_configs[0][0],
            edge_str=edge_str,
            node_str2=node_configs[1][0],
            filter_str=filter_addendum,
            node_return_str=node_addendum,
            edge_return_str=edge_return_str
        )
        
        return self.run_cypher(query_str, **kwargs)

    # Node creation methods

    @staticmethod
    def _normalize_value(value):
        if isinstance(value, str):
            return value.replace('"', '')
        
        return value

    def _normalize_attrs(self, attrs, string=True):
        attrs = {
            k: (
                'datetime({})'.format(self._normalize_datetime(v))
                if (is_datetime(v) or isinstance(v, pd.Timestamp) or v is pd.NaT) and v != 'M'
                else (self._normalize_value(v) if v is not None else 'null')
            ) for k, v in attrs.items()
        }
        
        return re.sub(
            r'"(?=:)', '',
            re.sub(r'(?<=, )"', '',
                re.sub(r'(?<="\))"', '',
                       re.sub(r'(?<=datetime\(null\))"', '',
                        re.sub(r'"(?=datetime)', '',
                            re.sub(r'\\"', '"',
                                re.sub(r'(?<=[{])"', '', json.dumps(attrs))
                              )
                        )
                    )
                )
            )
        ) if string else attrs

    def create_node(
        self,
        node_tag : str,
        node_attrs : dict,
        **kwargs
    ):
        query_str = '''
            MERGE (n:{node_tag} {node_attrs})
            RETURN n
        '''.format(
            node_tag=node_tag,
            node_attrs=self._normalize_attrs(node_attrs)
        )
        
        return self.run_cypher(query_str, **kwargs)

    # Edge creation methods

    def create_edge(
        self,
        node1 : dict,
        node2 : dict,
        edge_tag : str,
        edge_attrs : dict,
        **kwargs
    ):
        node1_tag = node1.pop('node_tag')
        node2_tag = node2.pop('node_tag')
        
        query_str = '''
            MATCH (n1:{node1_tag} {node1_attrs})
            MATCH (n2:{node2_tag} {node2_attrs})
            MERGE (n1) -[e:{edge_tag} {edge_attrs}]-> (n2)
            RETURN n1, n2, e
        '''.format(
            node1_tag=node1_tag,
            node1_attrs=self._normalize_attrs(node1),
            node2_tag=node2_tag,
            node2_attrs=self._normalize_attrs(node2),
            edge_tag=edge_tag,
            edge_attrs=self._normalize_attrs(edge_attrs)
        )

        return self.run_cypher(query_str, **kwargs)

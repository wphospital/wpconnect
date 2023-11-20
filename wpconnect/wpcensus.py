import pandas as pd
import requests
import re
from .query import Query

import os
from datetime import datetime



class Census:
    # url = 'http://10.16.8.21:1621/api/'
    url = 'http://127.0.0.1:5000/api/'
    full_output_cat_list = ['BusinessAndEconomy','Education','Employment','FamiliesAndLivingArrangements',
                 'Government','Health','Housing','IncomeAndPoverty','PopulationsAndPeople','RaceAndEthnicity']


    def __init__(
            self
    ):
        # self.url = 'http://10.16.8.21:1621/api/'
        # self.route = 'determine_file_source'
        # url = 'http://10.16.8.21:1621/api/'
        self.selected_output_cat_list = Census.full_output_cat_list
        self.patient_info = {}

    def call_api(
            self,
            route,
            params,
            return_type = 'json'
    ):
        """Calls CensusAPI at given url in 'url' class attribute

        Args:
            route (str): route to access at given api url
            params (dict): dictionary containing any set parameters for the API route
            return_type (str): Determines return format for API output ("json","full","dataframe"). Defaults to 'json'.

        Returns:
            resp/json/DataFrame: API ouptut formatted based on return_type set
        """

        #Make API call
        resp = requests.get(Census.url+route,params=params)
        # print(resp.json())


        if return_type == 'full':
            return resp
        elif route == 'demographic_data' or route == 'get_file_titles' or return_type == 'json':
            return resp.json()
        elif return_type.lower() == 'dataframe':
            return self.return_df(resp)
    
    def return_df(
            self,
            api_output
    ):
        """Converts api json output into a dataframe.
        Index is (geo_id,file_title).
        Columns of dataframe are column names returned from each file of api.
        Values are probabilities returned from each file (for the column returned from file)

        Args:
            api_output (response): output returned from census api

        Returns:
            dataframe: Dataframe representation of census api output.  Do not include files that did not return data for the given geo_id.
            File metadata and filtering on file are included in the dataframe
        """
        data = api_output.json()
        # data = api_output

        # print(data)

        df = pd.DataFrame(columns=["geo_id", "title"])
        df = df.set_index(['geo_id','title'])
        for id in data:
            #Set id_json to json for one location_id
            id_json = data[id]
            
            #For each file returned for the location
            for f in id_json:
                #If patient_info json, 
                if f == 'patient_info':
                    #Get dict of patient filters used for file (not None)
                    patient_info = {}
                    for info in id_json[f]:
                        if id_json[f][info] != None:
                            patient_info[info] = id_json[f][info]
                    
                    #Set patient info column for this id
                    # df.loc[id,f] = str(patient_info)
                    continue

                #Set file_json to json for one file at the location
                file_json = id_json[f]

                #Set attribute to json returned for file's value
                attribute = file_json['attribute']

                #If attribute returned is 'ID NOT FOUND' or 'File not valid for available patient data.', then don't add to dataframe
                if attribute == 'ID NOT FOUND' or attribute == 'File not valid for available patient data.':
                    continue
                #Else, add value to dataframe at multi-index [(location_id,file_title),column_name_returned]
                else:
                    file_title = str(file_json['metadata']['title'])
                    #Set metadata, title, and filtering for given file and given location_id
                    df.loc[(id,file_title),'metadata'] = str(file_json['metadata'])
                    # df.loc[(id,f),'title'] = file_title
                    if len(file_json['filtering on file']) >= 1:
                        df.loc[(id,file_title),'filtering'] = str(file_json['filtering on file'])

                    #Set value returned at [(location_id,file_title),column_name_returned]
                    # print(file_title,'---',list(attribute.values())[0])
                    if len(list(attribute.values())) > 1:
                        df.loc[(id,file_title),list(attribute.keys())[0]] = ','.join(map(str, list(attribute.values())))
                    else:
                        df.loc[(id,file_title),list(attribute.keys())[0]] = list(attribute.values())[0]#attribute[list(id_json['attribute'].keys())[0]]

            #Set patient_info for each file
            if id in df.index.get_level_values('geo_id'):
            # print(df.loc[(id,),:])
                df.loc[(id,),'patient_info'] = str(patient_info)
            # else:
            #     print(id)

        #Move patient_info to beginning of df
        cols = list(df.columns)
        if 'patient_info' in cols:
            cols.remove('patient_info')
            cols.insert(0,'patient_info')
        df = df.loc[:,cols]

        return df
    
    def return_df_for_id(self,df,id):
        id_index = id
        try:
            id_df = df.loc[(id_index,),:]
        except:
            return pd.DataFrame()
        # print(id_index)
        # id_df        
        # cols_to_check = list(id_df.columns)
        # cols_to_check.remove('patient_info')
        # cols_to_check.remove('metadata')
        
        return id_df
    
    def get_patient_info(self,mrn_list):
        self.patient_info = {}

        mrn_string = '\',\''.join(mrn_list)
        mrn_string = '\'' + mrn_string + '\''

        # print(os.getcwd())
        # print(os.path.join(os.pardir,'queries','patient_table.sql'))
        # # with open(os.path.join(os.getcwd(),'wpconnect','queries','patient_table.sql')) as file:
        # with open(os.path.join(os.pardir,'queries','patient_table.sql')) as file:
        # # with open('C:\\Users\\pseitz\\Documents\\wpconnect_branch\\wpconnect\\wpconnect\\queries\\patient_table.sql') as file:
        #     query = file.read()
        


        #Query wph database to get up to date patient information
        q = Query(connection_type="mit_edw",environ='qa',port=3923,username='pseitz',password='temppass!2023')
        query = q.import_sql('patient_table.sql')
        params_dict = {'mrn_list':mrn_string}
        query = query.format(**params_dict)
        patient_table = q.execute_query(query=query)

        patient_info = {}
        for row in patient_table.index:
            patient_row = patient_table.loc[row]

            # print(patient_row)

            #Create key in dict by patient's MRN
            patient_info[patient_row['mrn']] = []

            #Take first 5 digits of zipcode
            if patient_row['zipcode']:
                patient_info[patient_row['mrn']].append(patient_row['zipcode'].split('-')[0])
            else:
                patient_info[patient_row['mrn']].append(None)
            
            #Take censusblock value
            patient_info[patient_row['mrn']].append(patient_row['censusblock'])
            
            #Take patient age
            patient_info[patient_row['mrn']].append((pd.Timestamp.now() - patient_row['dob']) // pd.Timedelta(days=365.2425))

            #Patient sex
            patient_info[patient_row['mrn']].append(patient_row['sex'])

            #Paient employment status
            patient_info[patient_row['mrn']].append(patient_row['employmentstatusdesc'])

            #Patient ethnicity
            patient_info[patient_row['mrn']].append(re.sub(r'^[A-Z][0-9]\s','',patient_row['ethnicitydesc']))

            #Patient race
            patient_info[patient_row['mrn']].append(re.sub(r'^[A-Z][0-9]\s','',patient_row['racedesc']))

            self.patient_info = patient_info

        # print(patient_info)
        # print(self.patient_info)
        # return patient_info

    # def format_patient_dict(self,patient_dict):
    def format_patient_dict(self):
        self.pat_mrn = []
        self.id = []
        self.age = []
        self.sex = []
        self.employment_status = []
        self.ethnicity = []
        self.race = []
        
        # for mrn in patient_dict:
        for mrn in self.patient_info:
            self.pat_mrn.append(mrn)
            
            #id
            if self.patient_info[mrn][1]:
                self.id.append(self.patient_info[mrn][1])
            elif self.patient_info[mrn][0]:
                self.id.append(self.patient_info[mrn][0])
            else:
                self.id.append('-1')

            #age
            self.age.append(self.patient_info[mrn][2])

            #sex
            self.sex.append(self.patient_info[mrn][3])

            #employment_status
            # self.employment_status.append(patient_dict[mrn][4])
            employed_labels = ['Self Employed','Full Time','Student - Full Time','Part Time','Student - Part Time','On Active Military Duty']
            unemployed_labels = ['Retired','Not Employed','Disabled','Veteran''Former Employee']
            ignore_employment_labels = ['No Patient Contact','BLANK','On Leave','Unknown','UNKNOWN']
            if self.patient_info[mrn][4] in employed_labels:
                 self.employment_status.append(1)
            elif self.patient_info[mrn][4] in unemployed_labels:
                 self.employment_status.append(0)
            else:
                self.employment_status.append(-1)

            #ethnicity
            HL_labels = ['Bolivian','Puerto Rican','Guatemalan','Honduran','Latin American','Chilean','Hispanic or Latino','Nicaraguan','South American','Paraguayan',
                         'Peruvian','Venezuelan','Cuban','Panamanian','Argentinean','Colombian','Latin America','Mexican American','Mexicano','Ecuadorian','Spanish/Hispanic/Latino',
                         'Mexican','Hispanic-Central American','Costa Rican','Salvadoran','Uruguayan','Dominican']
            NHL_labels = ['Not Spanish/Hispanic/Latino',]
            ignore_ethnicity_labels = ['BLANK','Needs Clarification','Not Applicable/Unknown','Patient Unavailable','Patient Declined','UNKNOWN']
            if self.patient_info[mrn][5] in HL_labels:
                self.ethnicity.append(1)
            elif self.patient_info[mrn][5] in NHL_labels:
                self.ethnicity.append(0)
            else:
                self.ethnicity.append(-1)

            #race
            self.race.append(self.patient_info[mrn][6])

        return {'ids':self.id,'Patient Age':self.age,'Patient Sex':self.sex,'Patient Employment Status':self.employment_status,'Patient Ethnicity':self.ethnicity,'Patient Race':self.race}

    def add_api_output_to_df(self,enriched_df,api_output_df,output):
        """_summary_

        Args:
            enriched_df (Dataframe): Dataframe to enrich with API output
            api_output_df (Dataframe): Dataframe created from json ouput of census api /determine_file_source
        """

        now = datetime.now()

        added_cols = []
        #Determine which patient is which if two patients have same geo_id
        for returned_id in list(output.keys()):
            if '_' in returned_id:
                for pat in self.pat_mrn:
                    if returned_id.split('_')[0] in self.patient_info[pat]: 
                        if self.patient_info[pat][2] == output[returned_id]['patient_info']['AGE']:
                            self.patient_info[pat][self.patient_info[pat].index(returned_id.split('_')[0])] = returned_id
                            # print(self.patient_info[pat])
        
        

        now = datetime.now()
        #For each geo_id, get sub_df of geo_id data
        for mrn in self.pat_mrn:
            #If api returned data for given patient's geo_id (Checks zipcode then censusblockid), set id_df to contain data returned for only that geo_id
            if self.patient_info[mrn][0] in output.keys():
                id_df = self.return_df_for_id(api_output_df,self.patient_info[mrn][0])
            elif self.patient_info[mrn][1] and '1500000US' + self.patient_info[mrn][1] in output.keys():
                id_df = self.return_df_for_id(api_output_df,'1500000US' + self.patient_info[mrn][1])

            #Remove columns with no info
            id_df = id_df.loc[:,[not bool(re.fullmatch(r'Total((\s)*\.\d)?',col)) for col in id_df.columns]]

            # print(id_df)

            now_mid = datetime.now()


            #Can merge? But what about existing, new columns, they get _x and _y appended?  Is pd.merge faster?
            #For each file (row) in id_df
            for row in id_df.index:
                col_returned_for_file = id_df.loc[row,~id_df.loc[row,:].isna().values].index[-1]
                if col_returned_for_file != 'metadata' and col_returned_for_file != 'filtering':
                    if row not in added_cols:
                        added_cols.append(row)
                    # added_mrn.append(mrn)
                    #If not an aggregation file, can convert value to float.  Use those for testing later
                    try:
                        float(id_df.loc[row,~id_df.loc[row,:].isna().values][-1])
                        enriched_df.loc[enriched_df['mrn'] == mrn,row] = col_returned_for_file
                        # added_cols.append(row)
                    except ValueError:
                        enriched_df.loc[enriched_df['mrn'] == mrn,row] = id_df.loc[row,~id_df.loc[row,:].isna().values][-1]
                    
            end = datetime.now()
            total_time = (end-now_mid)
            print('Mid Step:',total_time)
        
        end = datetime.now()
        total_time = (end-now)
        print('Step 2:',total_time)

        return enriched_df, added_cols
    

    # def enrich_df(self,df,categories=None):
    def enrich_df(self,df):

        df = df.copy()

        if type(df) == pd.Series:
            mrn_list = df.values
        elif type(df) == pd.DataFrame:
            mrn_list = df['mrn'].values
        else:
            return pd.DataFrame()

        # if not categories:
        #     output_cat_list = ['BusinessAndEconomy','Education','Employment','FamiliesAndLivingArrangements',
        #          'Government','Health','Housing','IncomeAndPoverty','PopulationsAndPeople','RaceAndEthnicity']
        # else:
        #     output_cat_list = categories
        # if categories:
        #     self.selected_output_cat_list = categories
        
        completed_mrn = []
        for index in range(0,len(mrn_list),10):
            try:
                # patient_info = self.get_patient_info(mrn_list[index:index+10])
                self.get_patient_info(mrn_list[index:index+10])
                for pat in self.patient_info:
                    if self.patient_info[pat][1]:
                        self.patient_info[pat][1] = self.patient_info[pat][1][:-3]
                
                api_parameters = self.format_patient_dict()

                total_added_cols = []
                for cat in self.selected_output_cat_list:
                    api_parameters['output'] = cat

                    resp_json = self.call_api('determine_file_source',params=api_parameters,return_type='json')
                    api_df = self.call_api('determine_file_source',params=api_parameters,return_type='DataFrame')

                    new_pats, added_cols = self.add_api_output_to_df(df,api_df,resp_json)
                    total_added_cols.extend(added_cols)

                total_added_cols = list(set(total_added_cols))
                completed_mrn.extend(mrn_list[index:index+10])
            except:
                break

        #Just moved this out of the output_cat_list loop - RETEST TOMORROW
        return new_pats, total_added_cols, completed_mrn
        # new_pats.loc[new_pats['mrn'].isin(pat_list),:]
    
    # def enrich_df_for_model(self,df,categories=None):
    def enrich_df_for_model(self,df,interval=10):

        df = df.copy()

        #CHECK FOR UNIQUE MRN
        if type(df) == pd.Series and df.name.lower() == 'mrn':
            # mrn_list = df.values
            df = pd.DataFrame(df,columns=['mrn'])
            mrn_list = df['mrn'].values
        elif type(df) == pd.DataFrame:
            mrn_list = df['mrn'].values
        else:
            return pd.DataFrame()

        # if not categories:
        #     output_cat_list = ['BusinessAndEconomy','Education','Employment','FamiliesAndLivingArrangements',
        #          'Government','Health','Housing','IncomeAndPoverty','PopulationsAndPeople','RaceAndEthnicity']
        # else:
        #     output_cat_list = categories
        # if categories:
        #     self.selected_output_cat_list = categories
        
        block_and_zip_cols = self.call_api('get_file_titles',params={'folder':'BlockAndZip'})

        completed_mrn = []
        # interval = 80
        for index in range(0,len(mrn_list),interval):
            print('next round')
            # print(mrn_list[index:index+interval])
            try:
                # patient_info = self.get_patient_info(mrn_list[index:index+10])
                self.get_patient_info(mrn_list[index:index+interval])
                for pat in self.patient_info:
                    if self.patient_info[pat][1]:
                        self.patient_info[pat][1] = self.patient_info[pat][1][:-3]
                
                # api_parameters = self.format_patient_dict(patient_info)
                api_parameters = self.format_patient_dict()
                # api_parameters['override_cache'] = True
                
                api_parameters['output'] = 'All'

                # print(api_parameters)
                now = datetime.now()
                resp = self.call_api('determine_file_source',params=api_parameters,return_type='full')
                resp_json = resp.json()
                end = datetime.now()
                total_time = (end-now)
                print('API call 1:',total_time)
                
                now = datetime.now()
                # api_df = self.call_api('determine_file_source',params=api_parameters,return_type='DataFrame')
                # print(api_df.index)
                api_df = self.return_df(resp)
                end = datetime.now()
                total_time = (end-now)
                print('API call 2:',total_time)

                now = datetime.now()
                df, total_added_cols = self.add_api_output_to_df(df,api_df,resp_json)
                # total_added_cols.extend(added_cols)

                total_added_cols = list(set(total_added_cols))
                completed_mrn.extend(mrn_list[index:index+interval])
                end = datetime.now()
                total_time = (end-now)
                print('Add to df:',total_time)
            except Exception as e:
                print(e)
                break

        # try:
        #     block_and_zip_cols = self.call_api('get_file_titles',params={'folder':'BlockAndZip'})
        # except:
        #     block_and_zip_cols = 

        #In aggregation list also ????
        total_added_cols = [col for col in total_added_cols if col in block_and_zip_cols]

        # return new_pats, total_added_cols, completed_mrn
        return df, total_added_cols, completed_mrn
        # new_pats.loc[new_pats['mrn'].isin(pat_list),:]
    
    @staticmethod
    def get_full_cat_list():
        return Census.full_output_cat_list
    
    def get_selected_cat_list(self):
        return self.selected_output_cat_list
    
    def set_selected_cat_list(self,cat_list):
        self.selected_output_cat_list = cat_list
        
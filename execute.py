#Recommendation Model Creation Script :
def execute_model(client_id, user_id, process_id, upload_id, process_type, model_type, process_mode):
    #==================================================================
    # Importing Packages
    #==================================================================
    import os
    import pandas as pd
    import numpy as np
    import json
    import pickle
    import logging
    import vertica_python
    import mysql.connector
    from io import StringIO
    from datetime import datetime,timedelta
    from scipy.sparse import csr_matrix
    from matplotlib import pyplot as plt
    from sklearn.neighbors import NearestNeighbors

    #Function for fetching current date and time.
    def current_date_time():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    #==================================================================
    # Connecting with DB and Defining Paths
    #==================================================================
    #Reading environment variable
    acc_home = os.environ['ACCELERATOR_HOME']
    acc_home = acc_home+'/'
    with open(acc_home+'python_scripts/config.json','r') as json_file:
        configs = json.load(json_file)
    #Directory for fetching log file.
    Root = acc_home+configs['home']['client_home']
    path = "{}/{}/{}/{}.log".format(client_id,user_id,process_id,process_id)
    log_file = os.path.join(Root,path)

    # Defining Path
    root = acc_home+configs['home']['result']
    dirname = "{}/{}/{}".format(client_id, user_id, process_id)
    dirpath = os.path.join(root, dirname)

    #==========================================
    #Connecting to logger file
    #==========================================
    logging.basicConfig(filename=log_file, filemode='a+',format='%(asctime)s %(levelname)s %(funcName)s: %(lineno)d - %(message)s',datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    logger = logging
    log = True
    logger.info("Logger initialized successfully.")
    #adding analytics query file
    with open(acc_home+'python_scripts/analytics_stage_query.json', 'r') as json_file:
        query_config = json.load(json_file)
    with open(acc_home + 'python_scripts/DE_stage_query.json', 'r') as json_file:
        query_config = json.load(json_file)

    #Connection details for Vertica DB:
    conn_info = {'host':configs['config_args']['vertica']['host'],
    'user':configs['config_args']['vertica']['user'] ,
    'password':configs['config_args']['vertica']['password'],
    'database': configs['config_args']['vertica']['database'],
    'port': configs['config_args']['vertica']['port'],}

    #Connection details for mysql DB
    mysql_connection = mysql.connector.connect(host = configs['config_args']['mysql']['host'],
                          user = configs['config_args']['mysql']['user'],
                          passwd = configs['config_args']['mysql']['password'],
                          database = configs['config_args']['mysql']['database'])

    #User defined function for executing SQL queries.
    def mysql_execute(config_query,update_values):
        cursor = mysql_connection.cursor(prepared=True)
        cursor.execute(config_query,update_values)
        mysql_connection.commit()
        cursor.close()
    
    #Updating process status in process table 
    mysql_execute(query_config['UPDATE_PROCESS_CURRENT_STAGE'], ('Data Enrichment', 2, 3, current_date_time(), process_id))
    mysql_execute(query_config['UPDATE_PROCESS_COMPLETED_STEPS'], (2, current_date_time(), process_id))
    mysql_execute(query_config['INSERT_PROCESS_LOG'], (current_date_time(), 'Data Enrichment', 'Analytics', 0, process_id, current_date_time()))

    #Connecting to DB
    cursor = mysql_connection.cursor(prepared=True)
    logger.info("Successfully Connected to MySQL DB")
    #Reading out the model_id from process table
    cursor.execute(query_config['MODEL_ID_FETCH'],(process_id,))
    model_id = cursor.fetchall()[0][0]

    cursor.execute(query_config['BRAND_NAME_FETCH'],(process_id,))
    brand_name = cursor.fetchall()[0][0]

    if brand_name is None:
        brand_name = 'all'

    #reading out the model_execute_id from process table
    cursor.execute(query_config['MODEL_EXECUTION_FETCH'],(process_id,))
    model_execution_id = cursor.fetchall()[0][0]
    
    #reading out the graph path using model_id
    cursor.execute(query_config['GRAPH_PATH_FETCH'],(model_id,))
    model_id_path = cursor.fetchall()
    model_path = model_id_path[0][0]
    model_path = model_path + '/'

    if process_mode == "sftp":
        cursor.execute(query_config['MIN_DATE_FETCH_CLIENT'],(client_id,brand_name))
        start_date = cursor.fetchall()[0][0]
        if execution_date is None or execution_date == "" or len(str(execution_date).strip())==0:
            cursor.execute(query_config['MAX_DATE_FETCH_CLIENT'],(client_id,brand_name))
            end_date = cursor.fetchall()[0][0]
        else:
            end_date = str(datetime.strptime(execution_date, '%Y%m%d')-timedelta(days=1))

    elif int(upload_id) ==-1:
        cursor.execute(query_config['MIN_DATE_FETCH_PROCESS'],(process_id,))
        start_date = cursor.fetchall()[0][0]
        cursor.execute(query_config['MAX_DATE_FETCH_PROCESS'],(process_id,))
        end_date = cursor.fetchall()[0][0]
    else:
        cursor.execute(query_config['MIN_DATA_DATE_FETCH_PROCESS'],(upload_id,))
        start_date = cursor.fetchall()[0][0]
        cursor.execute(query_config['MAX_DATA_DATE_FETCH_PROCESS'],(upload_id,))
        end_date = cursor.fetchall()[0][0] 
           
    logger.info(start_date)
    logger.info(end_date)

    if brand_name.lower() == 'all':
        brand_name = ('%')
    #==================================================================
    # Reading Recommendation model input data
    #==================================================================
    #Reading Recco Rating data
    with open(acc_home+'sql_scripts/vertica_analytics_model_input_data_recco.sql', 'r') as sql_file:
        data_recco = sql_file.read()
    #Connecting to vertica db and fetching ratings data
    vertica_connection = vertica_python.connect(**conn_info)
    vsql_cur = vertica_connection.cursor()
    vsql_cur.execute(data_recco, {'process_id_var':upload_id,'client_id_var':client_id,'brand_name_var':brand_name,'start_date':start_date,'end_date':end_date})

    #Converting into DataFrame formate
    Rec = pd.DataFrame(vsql_cur.fetchall())
    logger.info("Fetching data from vertica db is completed.")
    Rec.columns = [c.name for c in vsql_cur.description]
    logger.info(list(Rec.columns))
    vertica_connection.close()


    #Reading Recco Rating data
    with open(acc_home+'sql_scripts/vertica_analytics_model_input_data_recco_ca_data.sql', 'r') as sql_file:
        ca_recco = sql_file.read()
    
    #Connecting to vertica db and fetching ratings data
    vertica_connection = vertica_python.connect(**conn_info)
    vsql_cur = vertica_connection.cursor()
    vsql_cur.execute(ca_recco, {'process_id_var':upload_id,'client_id_var':client_id,'brand_name_var':brand_name,'start_date':start_date,'end_date':end_date})

    #Converting into DataFrame formate
    Ca_data = pd.DataFrame(vsql_cur.fetchall())
    logger.info("Fetching data from vertica db is completed.")
    Ca_data.columns = [c.name for c in vsql_cur.description]
    logger.info(list(Ca_data.columns))
    vertica_connection.close()

    #Reading Recco Rating data
    with open(acc_home+'sql_scripts/vertica_analytics_model_input_data_recco_gt_data.sql', 'r') as sql_file:
        gt_recco = sql_file.read()
    
    #Connecting to vertica db and fetching ratings data
    vertica_connection = vertica_python.connect(**conn_info)
    vsql_cur = vertica_connection.cursor()
    vsql_cur.execute(gt_recco, {'process_id_var':upload_id,'client_id_var':client_id,'brand_name_var':brand_name,'start_date':start_date,'end_date':end_date})

    #Converting into DataFrame formate
    Gt_data = pd.DataFrame(vsql_cur.fetchall())
    logger.info("Fetching data from vertica db is completed.")
    Gt_data.columns = [c.name for c in vsql_cur.description]
    logger.info(list(Gt_data.columns))
    vertica_connection.close()

    #Updating Process and Process_log 
    mysql_execute(query_config['UPDATE_PROCESS_LOG'], (1, current_date_time(),0, 'Data Enrichment', 'Analytics', process_id))
    mysql_execute(query_config['UPDATE_PROCESS_COMPLETED_STEPS'], (3, current_date_time(), process_id))

    #==================================================================
    # Data Preparation
    #==================================================================
    # Updating Process Table
    mysql_execute(query_config['UPDATE_PROCESS_FIRST'],(0,3,"Analytics",current_date_time(), process_id))
    # Start Feature Engineering/Selection(process_log table)
    mysql_execute(query_config['INSERT_PROCESS_LOG'],(current_date_time(), "Analytics Layer", "Data Preparation", 0, process_id, current_date_time()))
    
    #Changing datatype 
    Rec['cust_ID'] = Rec.cust_ID.astype(str)
    Ca_data['customer_id'] = Ca_data.customer_id.astype(str)

    #Removing more than 25% missing values columns from data
    for i in Ca_data.columns.difference(['customer_Id']):
        if Ca_data[str(i)].isnull().sum() > (0.25 * Ca_data.shape[0]):
            Ca_data.drop(str(i), axis=1, inplace=True)

    for i in Gt_data.columns.difference(['Game_id']):
        if Gt_data[str(i)].isnull().sum() > (0.25 * Gt_data.shape[0]):
            Gt_data.drop(str(i), axis=1, inplace=True)

    # Seperating Numerical and categorical data type column names
    num_col = [key for key in dict(Ca_data.dtypes) if dict(Ca_data.dtypes)[key] in ['int64','int32','float64','float32']]
    cat_col = [key for key in dict(Ca_data.dtypes) if dict(Ca_data.dtypes)[key] in ['object','str']]
    Ca_Num = Ca_data[num_col]
    Ca_Cat = Ca_data[cat_col]

    #Imputing missing values 
    for i in list(Ca_Num.columns.difference(['customer_id'])):
        Ca_Num[str(i)] = Ca_Num[str(i)].fillna(Ca_Num[str(i)].median())

    for i in list(Ca_Cat.columns.difference(['customer_id'])):
        Ca_Cat[str(i)] = Ca_Cat[str(i)].fillna(Ca_Cat[str(i)].mode()[0])
    
    #Outlier treatment
    for col in Ca_Num.columns.difference(['customer_id']):
        percentiles = Ca_Num[col].quantile([0.01, 0.99]).values
        Ca_Num[col] = np.clip(Ca_Num[col], percentiles[0], percentiles[1])

    #Reducing number of categories for string columns
    dd = list(Ca_Cat.columns.difference(['customer_id']))
    kk = []
    for i in dd:
        if(len(Ca_Cat[str(i)].value_counts())) > 4:
            kk.append(i)
        
    Cat = pd.DataFrame()
    for i in kk:
        Val = Ca_Cat[str(i)].value_counts().index
        Val = Val[0:3]
        Cat[str(i)] = Val[0:3]
        Ca_Cat[str(i)] = np.where(Ca_Cat[str(i)] == Val[0], Val[0],np.where(Ca_Cat[str(i)] == Val[1], Val[1],np.where(Ca_Cat[str(i)] == Val[2], Val[2], 'Others')))

    # An utility function to create dummy variable
    def create_dummies(df, colname):
        col_dummies = pd.get_dummies(df[colname], prefix=colname, drop_first=True)
        df = pd.concat([df, col_dummies], axis=1)
        df.drop(colname, axis=1, inplace=True)
        return df

    for c_feature in dd:
        Ca_Cat[c_feature] = Ca_Cat[c_feature].astype('category')
        Ca_Cat = create_dummies(Ca_Cat, c_feature)

    # Combining both numerical and categorical data
    Ca_Cat.reset_index(drop=True, inplace=True)
    Ca_Num.reset_index(drop=True, inplace=True)
    Ca_data = pd.concat([Ca_Num, Ca_Cat], axis=1)

    # Symbols are not considered in Column names. Replacing with 2
    Ca_data.columns = Ca_data.columns.str.strip()
    Ca_data.columns = Ca_data.columns.str.replace(' ', '_')
    Ca_data.columns = Ca_data.columns.str.replace(r"[^a-zA-Z\d\_]+", "")
    Ca_data.columns = Ca_data.columns.str.replace(r"[^a-zA-Z\d\_]+", "")

    #Calculating quantile values for bet_qty,bet_amt and tot_hrs columns
    quantiles = Rec[['bet_qty','bet_amt','tot_hrs','recency']].quantile(q=[0.20, 0.40, 0.60, 0.80])
    quantiles = quantiles.to_dict()
    def FClass(x, p, d):
        if x <= d[p][0.20]:
            return 1
        elif x <= d[p][0.40]:
            return 2
        elif x <= d[p][0.60]:
            return 3
        elif x <= d[p][0.80]:
            return 4
        else:
            return 5
    
    def RClass(x, p, d):
        if x <= d[p][0.20]:
            return 1
        elif x <= d[p][0.40]:
            return 2
        elif x <= d[p][0.60]:
            return 3
        elif x <= d[p][0.80]:
            return 4
        else:
            return 5
    
    Rec['bet_qty_r'] = Rec['bet_qty'].apply(FClass, args=('bet_qty', quantiles,))
    Rec['bet_amt_r'] = Rec['bet_amt'].apply(FClass, args=('bet_amt', quantiles,))
    Rec['tot_hrs_r'] = Rec['tot_hrs'].apply(FClass, args=('tot_hrs', quantiles,))
    Rec['recency_r'] = Rec['recency'].apply(FClass, args=('recency', quantiles,))

    Rec['Tot_rating'] = round((Rec.bet_qty_r+Rec.bet_amt_r+Rec.tot_hrs_r+Rec.recency_r)/4,0)
    Rec.drop(['bet_qty_r','bet_amt_r','tot_hrs_r','bet_qty','bet_amt','tot_hrs','recency','recency_r'],axis=1,inplace=True)

    #Applied pivot table to get the data
    game_data = Rec.pivot(index='game_ID',columns='cust_ID',values='Tot_rating')
    cust_data = Rec.pivot(index='cust_ID',columns='game_ID',values='Tot_rating')
    #Handling null values
    game_data.fillna(0,inplace=True)
    cust_data.fillna(0,inplace=True)

    #Adding new columns
    game_data['Game_id'] = game_data.index
    cust_data['customer_id'] = cust_data.index

    #Changing data type of the column
    Ca_data['customer_id'] = Ca_data.customer_id.astype(str)

    #Merging all data
    gt_all_data = pd.merge(game_data,Gt_data, how='left', on='Game_id')
    ca_all_data = pd.merge(cust_data,Ca_data, how='left', on='customer_id')
    gt_all_data = gt_all_data.set_index('Game_id')
    ca_all_data = ca_all_data.set_index('customer_id')
    #Handling null values
    gt_all_data.fillna(0,inplace=True)
    ca_all_data.fillna(0,inplace=True)

    #Converting data into matrix form
    game_matrix = csr_matrix(gt_all_data.values)
    cust_matrix = csr_matrix(ca_all_data.values)

    #Reset the index values
    game_data.reset_index(inplace=True)
    cust_data.reset_index(inplace=True)

    #Updating process log table
    mysql_execute(query_config['UPDATE_PROCESS_LOG'],(1, current_date_time(),0,"Analytics Layer","Data Preparation", process_id))
    # Updating Process Table
    mysql_execute(query_config['UPDATE_PROCESS'],(1,'Running',current_date_time(), process_id))

    # Start Feature Engineering/Selection(process_log table)
    mysql_execute(query_config['INSERT_PROCESS_LOG'],(current_date_time(), "Analytics Layer", "Loading Model", 0, process_id, current_date_time()))

    #Performing Segmentation at game level and customer level
    knn_cust = pickle.load(open(model_path+'customer_segmentation.sav', 'rb'))
    knn_cust.fit(cust_matrix)
    knn_game = pickle.load(open(model_path+'games_segmentation.sav', 'rb'))
    knn_game.fit(game_matrix)

    #Updating process log table
    mysql_execute(query_config['UPDATE_PROCESS_LOG'],(1, current_date_time(),0,"Analytics Layer","Loading Model", process_id))
    # Updating Process Table
    mysql_execute(query_config['UPDATE_PROCESS'],(2,'Running',current_date_time(), process_id))

    # Start Feature Engineering/Selection(process_log table)
    mysql_execute(query_config['INSERT_PROCESS_LOG'],(current_date_time(), "Analytics Layer", "Result Prediction", 0, process_id, current_date_time()))

    #User defined function to get the similar games
    def get_game_recommendation(game_id,n_rec=10):
        game_list = Gt_data[Gt_data['Game_id'].str.contains(game_id)]  
        if len(game_list):  
            try:
                game_idx= game_list.iloc[0]['Game_id']
                game_idx = game_data[game_data['Game_id'] ==game_idx].index[0]
                distances , indices = knn_game.kneighbors(game_matrix[game_idx],n_neighbors=n_rec+1)    
                rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
                recommend_frame = []
                for val in rec_movie_indices:
                    game_idx = game_data.iloc[val[0]]['Game_id']
                    idx = Gt_data[Gt_data['Game_id'] == game_idx].index
                    Gt_data.iloc[idx]['Game_id'].values[0]
                    recommend_frame.append(Gt_data.iloc[idx]['Game_id'].values[0])
                return recommend_frame
            except:
                return "Data is not found"
        else:
            return "No movies found. Please check your input"

    #User defined function to get the similar customers
    def get_cust_recommendation(cust_id,n_rec=10):
        cust_list = Ca_data[Ca_data['customer_id'].str.contains(cust_id)]  
        if len(cust_list):
            try:
                cust_idx= cust_list.iloc[0]['customer_id']
                cust_idx = cust_data[cust_data['customer_id'] ==cust_idx].index[0]
                distances , indices = knn_cust.kneighbors(cust_matrix[cust_idx],n_neighbors=n_rec+1)
                rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
                recommend_frame = []
                for val in rec_movie_indices:
                    cust_idx = cust_data.iloc[val[0]]['customer_id']
                    idx = Ca_data[Ca_data['customer_id'] == cust_idx].index
                    recommend_frame.append(Ca_data.iloc[idx]['customer_id'].values[0])
                return recommend_frame
            except:
                return "Data is not found"
        else:
            return "No customers found. Please check your input"

    #User defined function for generating resommendations
    def recommendation_n(cust_id,n_rec=10,weight_knw=0.5,weight_recco=0.5):
        Player_knwn_games =list(np.unique(Rec[Rec.cust_ID ==cust_id]['game_ID']))
        
        knwn_games_list = list()
        for i in Player_knwn_games:
            knwn_games_list.append(i)
            similar_games = get_game_recommendation(i,int(n_rec))
            knwn_games_list = knwn_games_list+similar_games

        Player_similar_knwn_games = Rec[Rec['game_ID'].isin(knwn_games_list)]    
        Player_similar_knwn_games = Player_similar_knwn_games.groupby(['game_ID'],as_index=False)['Tot_rating'].mean()
        Player_similar_knwn_games = Player_similar_knwn_games.sort_values(['Tot_rating'],ascending=False)
        Player_similar_knwn_games['Tot_rating'] = Player_similar_knwn_games.Tot_rating
        Player_similar_knwn_games['known_games_yn'] = np.where(Player_similar_knwn_games['game_ID'].isin(Player_knwn_games),1,0)
        Player_similar_knwn_games = Player_similar_knwn_games.drop_duplicates(keep='first')
        similar_players = get_cust_recommendation(cust_id,int(n_rec))
        Rec_games = Rec[Rec['cust_ID'].isin(similar_players)]
        Rec_games = Rec_games.groupby(['game_ID'],as_index=False)['Tot_rating'].mean()
        Rec_games = Rec_games.sort_values(['Tot_rating'],ascending=False)
        Rec_games['Tot_rating'] = Rec_games.Tot_rating
        Rec_games['known_games_yn'] = 0
        Rec_games['known_games_yn'] = np.where(Rec_games['game_ID'].isin(Player_knwn_games),1,0)
        Rec_games = Rec_games.drop_duplicates('game_ID',keep='first')
        Data = pd.concat([Player_similar_knwn_games,Rec_games],axis=0).sort_values(['Tot_rating'],ascending=False)
        Data = Data.drop_duplicates('game_ID',keep='first')
        knw_games_list = list(Data[Data.known_games_yn==1]['game_ID'].head(int(n_rec/2)))
        if len(knw_games_list)==int(n_rec/2):
            n_rec_cnt = int(n_rec/2)
        else:
            n_rec_cnt = n_rec-len(knw_games_list)
        recc_games_list = list(Data[Data.known_games_yn==0]['game_ID'].head(n_rec_cnt))
        return knw_games_list+recc_games_list

    #List the players to get the recommendations
    K =list(np.unique(Rec.cust_ID.astype(str)))
    if brand_name == '%':
        brand_name = 'all'

    #Calculating recommendations for all players
    rec_result = pd.DataFrame(columns =['customer_id','recommended_games'])
    Pl_li = list()
    for i in K:
        try:
            Rec_input = 20
            Recco_list = recommendation_n(i,Rec_input)
            rec_result = rec_result.append({'customer_id':round(int(i),0),'recommended_games': ','.join(Recco_list)},ignore_index=True)
        except:
            Pl_li.append(i)

    #Adding all client related and model related information 
    rec_result['create_timestamp'] = datetime.now().strftime("%Y-%m-%d")
    rec_result['update_timestamp'] = datetime.now().strftime("%Y-%m-%d")
    rec_result['process_id'] = process_id
    rec_result['user_id'] = user_id
    rec_result['client_id'] = client_id
    rec_result['model_id'] = model_id
    rec_result['model_execution_id'] = model_execution_id
    rec_result['execution_date'] = datetime.now().strftime("%Y-%m-%d")
    rec_result['skin_code'] = brand_name
    rec_result['prediction_date'] = str(end_date)
    Rec_result = rec_result[['customer_id','recommended_games','create_timestamp','update_timestamp','process_id','user_id','client_id','model_id','model_execution_id','execution_date','skin_code','prediction_date']]

    buff = StringIO()
    # convert data frame to csv type
    for row in Rec_result.values.tolist():
        buff.write('{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n'.format(*row))
    #Storing results into vertica table
    vertica_connection = vertica_python.connect(**conn_info)  
    with vertica_connection.cursor() as cursor:
        cursor.copy(query_config['STORING_RESULTS'], buff.getvalue())
        vertica_connection.commit()
    vertica_connection.close()

    
    #Storing Players validation results in csv file
    logger.info('------------------------------')
    logger.info(Rec_result.shape)
    logger.info('------------------------------')
    Results_File = Rec_result[['customer_id','recommended_games','skin_code','execution_date']]
    Results_File.to_csv(dirpath+"/Prediction.csv",index=False)
    
    metrics_dict = {"prediction_date": str(end_date),"brand_name" : str(brand_name)}
    metrics_dict =json.dumps(metrics_dict)

    #Updating model details in model table
    mysql_execute(query_config['UPDATE_MODEL_EXECUTION_TABLE'],(len(list(Rec_result.customer_id)),metrics_dict,'Done',dirpath,model_id,user_id,current_date_time(),model_execution_id))

    #Updating process log table
    mysql_execute(query_config['UPDATE_PROCESS_LOG'],(1, current_date_time(),0,"Analytics Layer","Result Prediction", process_id))
    # Updating Process Table
    mysql_execute(query_config['UPDATE_PROCESS'],(3,'Done',current_date_time(), process_id))



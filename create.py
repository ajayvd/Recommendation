#Recommendation Model Creation Script :
def create_model(client_id, user_id, process_id, upload_id, process_type, model_type, process_mode):
    #==================================================================
    # Importing Packages
    #==================================================================
    import os
    import pandas as pd
    import numpy as np
    import json
    import pickle
    import plotly
    import logging
    import vertica_python
    import mysql.connector
    import plotly.graph_objects as go
    from datetime import datetime,timedelta
    from scipy.sparse import csr_matrix
    from matplotlib import pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    from matplotlib import gridspec

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
    end_date = end_date - timedelta(days=30)
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

    #Creating customer_id tuple to fetch validation data
    cust_id_tup = tuple(np.unique(Rec.cust_ID).astype(str))

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

    #Reading Recco Validation data
    with open(acc_home+'sql_scripts/vertica_analytics_recco_validation_data.sql', 'r') as sql_file:
        gt_recco = sql_file.read()
    
    #Connecting to vertica db and fetching ratings data
    vertica_connection = vertica_python.connect(**conn_info)
    vsql_cur = vertica_connection.cursor()
    vsql_cur.execute(gt_recco, {'process_id_var':upload_id,'client_id_var':client_id,'brand_name_var':brand_name,'start_date':start_date,'end_date':end_date,'cust_id_tup':cust_id_tup})

    #Converting into DataFrame formate
    val_data = pd.DataFrame(vsql_cur.fetchall())
    logger.info("Fetching data from vertica db is completed.")
    val_data.columns = [c.name for c in vsql_cur.description]
    logger.info(list(val_data.columns))
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
    mysql_execute(query_config['INSERT_PROCESS_LOG'],(current_date_time(), "Analytics Layer", "Model Training", 0, process_id, current_date_time()))

    #Performing Segmentation at game level and customer level
    knn_cust = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn_cust.fit(cust_matrix)
    knn_game = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn_game.fit(game_matrix)

    #Saving Customer Segmentation and Games Segmentation models
    filename = "/customer_segmentation.sav"
    pickle.dump(knn_cust, open(dirpath+filename, 'wb'))
    filename = "/games_segmentation.sav"
    pickle.dump(knn_game, open(dirpath+filename, 'wb'))

    #Updating process log table
    mysql_execute(query_config['UPDATE_PROCESS_LOG'],(1, current_date_time(),0,"Analytics Layer","Model Training", process_id))
    # Updating Process Table
    mysql_execute(query_config['UPDATE_PROCESS'],(2,'Running',current_date_time(), process_id))

    
    # Start Feature Engineering/Selection(process_log table)
    mysql_execute(query_config['INSERT_PROCESS_LOG'],(current_date_time(), "Analytics Layer", "Model Testing", 0, process_id, current_date_time()))

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
    
    val_data['customer_id'] = val_data['customer_id'].astype(str)

    #List the players to get the recommendations
    K =list(np.unique(val_data.customer_id.astype(str)))
    K = K[0:500]
    
    #Performing recommendation for all players and calculating metrics to validate the recommendations
    rec_result = pd.DataFrame(columns =['customer_id','Total_games_played (P)','Recco_games_cnt (Q)','Recco_known_games_cnt (R)',
                                        'Recco_new_games_cnt (S)','Played_recco_game_cnt (A=B+C)','Played_recco_known_games_cnt (B)','Played_recco_new_games_cnt (C)',
                                        'Acc_wrto_recco_all (A/Q)','Acc_wrto_known_games (B/R)','Acc_wrto_new_games (C/S)','Overall_Accuracy (A/P)'])
    Pl_li = list()
    for i in K:
        try:
            Rec_input = 20
            Recco_list = recommendation_n(i,Rec_input)
            Recco_list = set(Recco_list)
            Val_list = set(val_data[val_data.customer_id == str(i)]['game_id'].value_counts().index)
            Knwn_list =set(Rec[Rec.cust_ID == i]['game_ID'].value_counts().index)
            matching_recco_knwn = set(Knwn_list & Recco_list)
            mat_wrto_knwn_games = len(matching_recco_knwn & Val_list)
            mat_overall_games = len(Recco_list & Val_list)
            mat_wrto_new_games = mat_overall_games - mat_wrto_knwn_games
            #matching_recco_games = len(Recco_list & Val_list)
            rec_result = rec_result.append({'customer_id':round(int(i),0),'Total_games_played (P)':int(len(Val_list)),'Recco_games_cnt (Q)':int(Rec_input),
                           'Recco_known_games_cnt (R)':int(len(matching_recco_knwn)),'Recco_new_games_cnt (S)':int(Rec_input-len(matching_recco_knwn)),
                           'Played_recco_game_cnt (A=B+C)':int(mat_overall_games),'Played_recco_known_games_cnt (B)':int(mat_wrto_knwn_games),
                           'Played_recco_new_games_cnt (C)':mat_wrto_new_games,'Acc_wrto_recco_all (A/Q)':round((mat_overall_games/Rec_input)*100,2) if mat_overall_games>0 else 0,
                           'Acc_wrto_known_games (B/R)':round((mat_wrto_knwn_games/len(matching_recco_knwn))*100,2) if mat_wrto_knwn_games>0 else 0,
                           'Acc_wrto_new_games (C/S)':round((mat_wrto_new_games/(Rec_input-len(matching_recco_knwn)))*100,2) if mat_wrto_new_games>0 else 0,
                           'Overall_Accuracy (A/P)': round((int(mat_overall_games)/int(len(Val_list)))*100,2) if mat_overall_games>0 else 0},ignore_index=True)
        except:
            Pl_li.append(i)

    #===================================================
    # Plotting Model validation results
    #===================================================

    def bar_plot(col_name,plt_name):
        facecolor = '#EAEAEA'
        color_bars = '#3475D0'
        txt_color1 = '#252525'
        txt_color2 = '#004C74'
        fig, ax = plt.subplots(1, figsize=(7,4.25), facecolor=facecolor)
        ax.set_facecolor(facecolor) 
        n, bins, patches = plt.hist(rec_result[str(col_name)], color=color_bars, bins='doane')
        #grid
        minor_locator = AutoMinorLocator(2)
        plt.gca().xaxis.set_minor_locator(minor_locator)
        plt.grid(which='minor', color=facecolor, lw = 0.5)
        xticks = [(bins[idx+1] + value)/2 for idx, value in enumerate(bins[:-1])]
        xticks_labels = [ "{:.0f}-{:.0f}".format(value if value > 0 else 0, bins[idx+1]) for idx, value in enumerate(bins[:-1])]
        plt.xticks(xticks, labels=xticks_labels, c=txt_color1, fontsize=13)
        # remove major and minor ticks from the x axis, but keep the labels
        ax.tick_params(axis='x', which='both',length=0)
        # Hide the right and top spines
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for idx, value in enumerate(n):
            if value > 0:
                plt.text(xticks[idx], value+0.1, int(value), ha='center', fontsize=10, c=txt_color1)
        plt.xlabel('Accuracy Buckets(%)', c=txt_color2, fontsize=10)
        plt.xticks(rotation =40)
        plt.ylabel('Player Count', c=txt_color2, fontsize=10)
        plt.tight_layout()
        plt.savefig(dirpath+plt_name, facecolor=facecolor)

    def bar_plt(col_name, plt_name):
        cut_labels_4 = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100']
        cut_bins = [0,10,20,30,40,50,60,70,80,90,100]
        rec_result['cut_ex1'] = pd.cut(rec_result[str(col_name)], bins=cut_bins, labels=cut_labels_4,include_lowest=True)
        I = list(rec_result.cut_ex1.value_counts().sort_index().index)
        V = list(rec_result.cut_ex1.value_counts().sort_index().values)
        facecolor = '#EAEAEA'
        color_bars = '#3475D0'
        txt_color1 = '#252525'
        txt_color2 = '#004C74'
        plt.subplots(1, figsize=(7,4.25),facecolor=facecolor)
        plt.grid(which='minor', color=facecolor, lw = 0.5)
        plt.bar(I,V)    
        for i in range(len(V)):
            plt.annotate(str(V[i]), xy=(I[i],V[i]), ha='center', va='bottom')
        plt.xticks(rotation =35)
        plt.ylabel('Player Count', c=txt_color2, fontsize=10)
        plt.xlabel('Accuracy Buckets(%)', c=txt_color2, fontsize=10)
        plt.savefig(dirpath+plt_name, facecolor=facecolor, bbox_inches='tight', dpi=100)

    #Plotting for values
    for i,j in zip(['Overall_Accuracy (A/P)','Acc_wrto_recco_all (A/Q)','Acc_wrto_known_games (B/R)','Acc_wrto_new_games (C/S)'],['/Total_Played_Games_accuracy.png','/Recommended_games_accuracy.png','/Known_games_recommended_accuracy.png','/New_games_recommended_accuracy.png']):
        try:
            bar_plt(i,j)
        except Exception as ex:
            logger.info(ex)

    '''
    config = {'displayModeBar': False}
    try:
        #Plotting Overall accuracy values
        fig = go.Figure(go.Histogram(x=rec_result['Overall_Accuracy (A/P)'],xbins=dict(start=0,end=100,size=10),bingroup=1))

        fig.update_layout(bargap=0.01, barmode="overlay",height=450,width=500, 
            xaxis_title_text='Accuracy Buckets (%)', yaxis_title_text='Player Count')
        plotly.offline.plot(fig, filename=dirpath+'/Total_Played_Games_accuracy.html',config=config)

        #Plotting accuracy values wrto Recommended games
        fig = go.Figure(go.Histogram(x=rec_result['Acc_wrto_recco_all (A/Q)'],xbins=dict(start=0,end=100,size=10),bingroup=1))

        fig.update_layout(bargap=0.01, barmode="overlay",height=450,width=500, 
            xaxis_title_text='Accuracy Buckets (%)', yaxis_title_text='Player Count')
        plotly.offline.plot(fig, filename=dirpath+'/Recommended_games_accuracy.html',config=config)

        #Plotting accuracy values wrto known games
        fig = go.Figure(go.Histogram(x=rec_result['Acc_wrto_known_games (B/R)'],xbins=dict(start=0,end=100,size=10),bingroup=1))

        fig.update_layout(bargap=0.01, barmode="overlay",height=450,width=500, 
            xaxis_title_text='Accuracy Buckets (%)', yaxis_title_text='Player Count')
        plotly.offline.plot(fig, filename=dirpath+'/Known_games_recommended_accuracy.html',config=config)

        #Plotting accuracy values wrto new games
        fig = go.Figure(go.Histogram(x=rec_result['Acc_wrto_new_games (C/S)'],xbins=dict(start=0,end=100,size=10),bingroup=1))
        fig.update_layout(bargap=0.01, barmode="overlay",height=450,width=500, 
            xaxis_title_text='Accuracy Buckets (%)', yaxis_title_text='Player Count')
        plotly.offline.plot(fig, filename=dirpath+'/New_games_recommended_accuracy.html',config=config)
    except Exception as er:
        logger.info(er)
        '''



    #Checking validation for the players who have played more than 2 games in validation time period
    rec_result['Error_Square'] =  (Rec_input - rec_result['Played_recco_game_cnt (A=B+C)'])*(Rec_input - rec_result['Played_recco_game_cnt (A=B+C)'])
    RMSE = np.sqrt(rec_result.Error_Square.sum())
    Recco_validation = rec_result.T
    
    #Storing Players validation results in csv file
    Recco_validation.to_csv(dirpath+"/Recco_Validation_Report.csv",index=False)
    
    #Updating model details in model table
    #Storing Accuracy value in model class metric table
    logger.info("Storing Time Period value in model class metric table")
    mysql_execute(query_config['INSERT_RESULTS_IN_CLASS_METRICS'],("Test","Accuracy",np.round(rec_result['Acc_wrto_recco_all (A/Q)'].mean(),2), int(model_id)))
    mysql_execute(query_config['UPDATE_MODEL_TABLE'],(np.round(rec_result['Acc_wrto_recco_all (A/Q)'].mean(),2),len(np.unique(Rec['cust_ID'])),current_date_time(),dirpath,'Done',current_date_time(),model_id))

    #Updating process log table
    mysql_execute(query_config['UPDATE_PROCESS_LOG'],(1, current_date_time(),0,"Analytics Layer","Model Testing", process_id))
    # Updating Process Table
    mysql_execute(query_config['UPDATE_PROCESS'],(3,'Done',current_date_time(), process_id))




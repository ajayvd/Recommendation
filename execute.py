#Responsible Gambling Model Execution Script :
def execute_model(client_id,user_id,process_id,upload_id,model_id,model_type,process_mode):
  # ==================================================================
  # Importing Packages
  # ==================================================================
    import os
    import matplotlib
    matplotlib.use('Agg')
    import pandas as pd
    import numpy as np
    import pickle
    import logging
    import sys
    from io import StringIO
    import json
    import shutil
    import vertica_python
    import mysql.connector
    from matplotlib import pyplot as plt
    from datetime import datetime
    from matplotlib.ticker import MaxNLocator
    import warnings
    warnings.filterwarnings('ignore')
    #Function for fetching current date and time.
    def current_date_time():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #==================================================================
    # Connecting with DB
    #==================================================================
    #Reading Environment variable
    acc_home = os.environ['ACCELERATOR_HOME']
    acc_home = acc_home+'/'
    with open(acc_home+'python_scripts/config.json','r') as json_file:
        configs = json.load(json_file)
    #Directory for fetching log file.
    Root = acc_home+configs['home']['client_home']
    path = "{}/{}/{}/{}.log".format(client_id,user_id,process_id,process_id)
    log_file = os.path.join(Root,path)
    #==========================================
    #Connecting to logger file
    #==========================================
    logging.basicConfig(filename=log_file, filemode='a+', format='%(asctime)s %(levelname)s %(funcName)s: %(lineno)d - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    logger = logging
    log = True
    logger.info("Logger initialized successfully.")
    #adding analytics query file
    with open(acc_home+'python_scripts/analytics_stage_query.json', 'r') as json_file:
        query_config = json.load(json_file)
    with open(acc_home+'sql_scripts/vertica_schema_set.sql', 'r') as sql_file:
        schema_set_query = sql_file.read()
    schema_qualifier = os.environ['SCHEMA_QUALIFIER']
    
    #using the env variable to switch to dev schemas here
    schema_type = configs['schema_type'][schema_qualifier]
    stage_schema = schema_type['stage']
    dw_schema = schema_type['dw']
    analytics_schema = schema_type['analytics']
    #=>Connection Details:
    conn_info = {'host':configs['config_args']['vertica']['host'],
    'user':configs['config_args']['vertica']['user'] ,
    'password':configs['config_args']['vertica']['password'],
    'database': configs['config_args']['vertica']['database'],  
    'port': configs['config_args']['vertica']['port'],}
    #Connecting to the DB
    vertica_connection = vertica_python.connect(**conn_info)
    vsql_cur = vertica_connection.cursor()
    logger.info("Successfully connected to Vertica DB")
    #Query for extracting data to particular Process_Id
    process_id_var = str(process_id)
    client_id_var = str(client_id)
    # Executing the statement
    vsql_cur.execute(schema_set_query, {'schema_name': analytics_schema})
    vsql_cur.execute(query_config['DATA_QUERY']['RG_EXECUTE'],{'process_id_var': process_id_var, 'client_id_var': client_id_var})
    #Converting into DataFrame formate
    Validation = pd.DataFrame(vsql_cur.fetchall())
    logger.info("Fetching data from Vertica DB is Completed")
    Validation.columns = [c.name for c in vsql_cur.description]
    vertica_connection.close()

    #Connection detils for MySQL DB
    mysql_connection = mysql.connector.connect(host = configs['config_args']['mysql']['host'],
                          user = configs['config_args']['mysql']['user'],
                          passwd = configs['config_args']['mysql']['password'],
                          database = configs['config_args']['mysql']['database'])
    cursor = mysql_connection.cursor(prepared=True)
    logger.info("Successfully conneted to MySQL DB")
    #reading out the model_execute_id from process table
    cursor.execute(query_config['MODEL_EXECUTION_FETCH'],(process_id,))
    model_execution_id = cursor.fetchall()
    model_execution_id = model_execution_id[0][0]
        
    #reading out the graph path from model_id
    cursor.execute(query_config['GRAPH_PATH_FETCH'],(model_id,))
    model_id_path = cursor.fetchall()
    model_path = model_id_path[0][0]
    model_path = model_path + '/'

    #Brand_name and Prediction date fetch from process table
    if process_mode != "sftp":
        cursor.execute(query_config['BRAND_NAME_FETCH_MODEL'],(model_id,))
        brand_name = cursor.fetchall()[0][0]
        brand_name = str(brand_name)
        logger.info(brand_name)
        if brand_name == "None":
            brand_name = 'all'
        cursor.execute(query_config['PREDICTION_DATE_YN'],(process_id,))
        prediction_date_yn = cursor.fetchall()[0][0]
        if int(prediction_date_yn) == 1:
            cursor.execute(query_config['MAX_DATE_FETCH_PROCESS'],(process_id,))
            end_date = cursor.fetchall()[0][0]
        else:
            cursor.execute(query_config['MAX_DATA_DATE_FETCH_PROCESS'],(upload_id,))
            end_date = cursor.fetchall()[0][0]
    else:
        cursor.execute(query_config['BRAND_NAME_FETCH_MODEL'],(model_id,))
        brand_name = cursor.fetchall()[0][0]
        brand_name = str(brand_name)
        logger.info(brand_name)
        if brand_name == "None":
            brand_name = 'all'
        cursor.execute(query_config['MAX_DATE_FETCH_PROCESS'],(process_id,))
        end_date = cursor.fetchall()[0][0]

    #User defined function for executing SQL queries.
    def mysql_execute(config_query,update_values):
        cursor = mysql_connection.cursor(prepared=True)
        cursor.execute(config_query,update_values)
        mysql_connection.commit()
        cursor.close()
    #Less number of Samples in training model
    if int(Validation.shape[0]) < 1:
        logger.info("No Data for Model Execution")
        mysql_execute(query_config['UPDATE_ERROR_IN_MODEL_EXECUTION_TABLE'],("Error",current_date_time(),model_execution_id))
        sys.exit()

    #Dropping duplicates from data
    Validation = Validation.drop_duplicates(['customer_id'])

    #Seperating Harm Markers data
    harm_Markers_data = Validation[['customer_id','diff_in_spend_m1', 'hours_spent_on_site_m2', 'freq_play_increase_m3', 'deposit_frequency_m4', 'declined_deposit_m5', 'cancelled_withdrawls_m8', 'hours_spent_late_night_m9']]
    harm_Markers_data.columns = ['customer_id','IADD', 'HDATS', 'IDATS', 'HDF', 'HDD', 'HCW', 'HLNA']
    Harm_Markers_data = harm_Markers_data[harm_Markers_data.columns.difference(['customer_id'])].round(2)
    Harm_Markers_data['customer_id'] = harm_Markers_data.customer_id

    #Deleting harm markers data
    Validation = Validation.drop(['diff_in_spend_m1', 'hours_spent_on_site_m2', 'freq_play_increase_m3', 'deposit_frequency_m4', 'declined_deposit_m5', 'cancelled_withdrawls_m8', 'hours_spent_late_night_m9'],axis=1)

    #==================================================================
    # Data Preparation :
    #==================================================================
    logger.info("Data Preparation is Completed")
    #Updating Process Table
    mysql_execute(query_config['UPDATE_PROCESS_FIRST'],(0,3,"Analytics",current_date_time(),process_id))
    #Updating Process_log table
    mysql_execute(query_config['INSERT_PROCESS_LOG'],(current_date_time(), "Analytics Layer",0,"Data Preparation",current_date_time(),process_id))

    #More Missing values in test data.
    if Validation.isnull().sum().sum() > (int(Validation.shape[0]*Validation.shape[1]*0.35)):
        logger.info("More number of missing values in data")
    #Defining Target variable
    Validation['Target2']=np.where(Validation.total_score < Validation.total_score.quantile(0.50),0,np.where(Validation.total_score < Validation.total_score.quantile(0.85),1,np.where(Validation.total_score < Validation.total_score.quantile(0.95),2,3)))
    Validation.drop(['total_score'],axis=1,inplace=True)

    #Dropping Variables which have more than 35% missing values
    Drop_Var = pd.read_csv(model_path+"Droped_Var.csv")
    if Drop_Var.shape[0]>0:
        Validation.drop(list(Drop_Var.Columns),axis=1,inplace=True)

    #Seperating Numerical and categorical data type column names
    num_col=[key for key in dict(Validation.dtypes) if dict(Validation.dtypes)[key] in ['int64','int32','float64','float32']]
    cat_col=[key for key in dict(Validation.dtypes) if dict(Validation.dtypes)[key] in ['object']]
    #Seperating Numerical and categorical variables
    Validation_Num = Validation[num_col]
    Validation_Cat = Validation[cat_col]
    #Imputing missing values:
    loaded_missing = pickle.load(open(model_path+'Missing_Impution_Object.sav', 'rb'))
    Validation_Num=pd.DataFrame(loaded_missing.transform(Validation_Num[Validation_Num.columns]),columns=Validation_Num.columns)
    Validation_Num.index = Validation_Cat.index

    # #### Outlier Treatment :
    for col in Validation_Num.columns.difference(['customer_id']):
        percentiles = Validation_Num[col].quantile([0.01,0.99]).values
        Validation_Num[col] = np.clip(Validation_Num[col], percentiles[0], percentiles[1])

    #Reading Categories from Categorical table
    Categories = pd.read_csv(model_path+"Categories_Res.csv")
    Categories = Categories.iloc[:,1:]

    #Reducing Categories
    for i in Categories.columns:
        Val= list(Categories[str(i)].values)
        Validation_Cat[str(i)]=np.where(Validation_Cat[str(i)]==Val[0],Val[0],
                                        np.where(Validation_Cat[str(i)]==Val[1],Val[1],
                                                 np.where(Validation_Cat[str(i)]==Val[2],Val[2],'Others')))

    #Imputing Missing values with Mode Value
    for i in Validation_Cat.columns.difference(['customer_id','Target2']):
        Validation_Cat.fillna(Validation_Cat[str(i)].mode()[0],inplace=True)
    # An utility function to create dummy variable
    def create_dummies( df, colname ):
        col_dummies = pd.get_dummies(df[colname], prefix=colname)
        df = pd.concat([df, col_dummies], axis=1)
        df.drop( colname, axis = 1, inplace = True )
        return df

    #Creating dummy variable for categorical variables
    for c_feature in Validation_Cat.columns.difference(['Target2','customer_id']):
        Validation_Cat[c_feature] = Validation_Cat[c_feature].astype('category')
        Validation_Cat = create_dummies(Validation_Cat , c_feature )

    #Combining both numerical and categorical data
    Validation_Num.reset_index(drop=True, inplace=True)
    Validation_Cat.reset_index(drop=True, inplace=True)
    Validation = pd.concat([Validation_Num,Validation_Cat],axis=1)
    #Symbols are not considered in Column names. Replacing with 2
    Validation.columns = Validation.columns.str.strip()
    Validation.columns = Validation.columns.str.replace(' ', '_')
    Validation.columns = Validation.columns.str.replace(r"[^a-zA-Z\d\_]+", "")
    Validation.columns = Validation.columns.str.replace(r"[^a-zA-Z\d\_]+", "")

    #Reading All selected variables
    with open(model_path+"Features_RG2.bin", "rb") as data:
        Features_RG2 = pickle.load(data)

    #Checking selected features
    for i in list(Features_RG2):
        if i not in list(Validation.columns):
            Validation[str(i)] = 0

    #Filtering Important features
    Test_X = Validation[Features_RG2]

    #Update 'Data Preparation completed'
    mysql_execute(query_config['UPDATE_PROCESS_LOG'],(1,current_date_time(),"Data Preparation",process_id))
    #Updating Process Table
    mysql_execute(query_config['UPDATE_PROCESS'],(1,'Running',current_date_time(),process_id))
    logger.info("Data Preparation is Completed")
    #==================================================================
    # Loading Model
    #==================================================================
    logger.info("Loading Model is Started")
    #Updating Loading model status
    mysql_execute(query_config['INSERT_PROCESS_LOG'],(current_date_time(), "Analytics Layer",0,"Loading Model",current_date_time(),process_id))

    # load the model from disk
    loaded_model2 = pickle.load(open(model_path+'Finalized_Model_Res2.sav', 'rb'))

    #Update 'Loading Model completed'
    mysql_execute(query_config['UPDATE_PROCESS_LOG'],(1,current_date_time(),"Loading Model",process_id))
    #Updating Process Table
    mysql_execute(query_config['UPDATE_PROCESS'],(2,'Running',current_date_time(),process_id))
    logger.info("Loading Model is Completed")
    #==================================================================
    # Result Prediction
    #==================================================================
    logger.info("Result Prediction is Started")
    #Updating Resuilt Prediction status
    mysql_execute(query_config['INSERT_PROCESS_LOG'],(current_date_time(), "Analytics Layer",0,"Result Prediction",current_date_time(),process_id))

    #Creating path for storing execute results
    root = acc_home+configs['home']['result']
    dirname = "{}/{}/{}/".format(client_id,user_id,process_id)
    r_dirpath = os.path.join(root,dirname)

    #Predicting values using data
    Predictions = pd.DataFrame(loaded_model2.predict_proba(Test_X))
    Predictions.columns = ['Prob_0','Prob_1','Prob_2','Prob_3']
    Predictions['Pred'] = loaded_model2.predict(Test_X)

    #Validating prediction results
    if (len(Predictions['Prob_0'].value_counts()) == 1) and (len(Predictions['Prob_1'].value_counts()) == 1) and (len(Predictions['Prob_2'].value_counts()) == 1):
        #Updating AL_3 error in error_log table
        mysql_execute(query_config['INSERT_ERROR_LOG'],(current_date_time(),"AL_3",process_id))
        #Updating error in notification table about :
        mysql_execute(query_config['UPDATE_NOTIFICATION'],(current_date_time(),"Error",0,2,0,0,process_id))
        #Updating Error in Model Execution table
        mysql_execute(query_config['UPDATE_ERROR_IN_MODEL_EXECUTION_TABLE'],("Error",current_date_time(),model_execution_id))
        sys.exit()

    Predictions['Customer_Id'] = Validation.customer_id
    Predictions = Predictions[['Customer_Id','Pred','Prob_0','Prob_1','Prob_2','Prob_3']]
    Predictions['Prob_0'] = np.round(Predictions.Prob_0,2)
    Predictions['Prob_1'] = np.round(Predictions.Prob_1,2)
    Predictions['Prob_2'] = np.round(Predictions.Prob_2,2)
    Predictions['Prob_3'] = np.round(Predictions.Prob_3,2)
    Predictions['Client_Id'] = client_id
    Predictions['User_Id'] = user_id
    Predictions['Process_Id'] = process_id

    #Saving Probability Plot
    plt.close('all')
    Predictions2= pd.DataFrame()
    Predictions2['Pred'] = np.where(Predictions.Pred ==0,"No Risk",np.where(Predictions.Pred ==1,"Low Risk",np.where(Predictions.Pred==2,"Medium Risk","High Risk")))
    Ply_ind = list(Predictions2.Pred.value_counts().index)
    Ply_val = list(Predictions2.Pred.value_counts().values)
    plt.figure(figsize=(5.5, 5))
    plt.bar(Ply_ind,Ply_val,)
    plt.title("Predicted Player Count")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Risk Classes")
    plt.ylabel("Player Count")
    plt.savefig(r_dirpath+'Probabilty_Churn_Plot.png')

    Predictions4 = Predictions[['Customer_Id','Pred','Prob_0','Prob_1','Prob_2','Prob_3','Client_Id','User_Id','Process_Id']]
    Predictions4['Pred'] = np.where(Predictions.Pred ==0,"No Risk",np.where(Predictions.Pred ==1,"Low Risk",np.where(Predictions.Pred==2,"Medium Risk","High Risk")))
    Predictions4 = Predictions4[['Customer_Id','Pred','Client_Id','User_Id','Process_Id']]
    Predictions4 = Predictions4.dropna(subset=['Customer_Id'])
    logger.info("Results Prediction is Completed")
    Predictions4['create_timestamp'] = current_date_time()
    Predictions4['update_timestamp'] = current_date_time()
    Predictions4= pd.merge(Predictions4,Harm_Markers_data, left_on='Customer_Id', right_on='customer_id', how='left')
    Predictions4.drop(['customer_id'],axis=1,inplace=True)
    Predictions4 = Predictions4[['Customer_Id','Pred','Client_Id','User_Id','Process_Id','create_timestamp','update_timestamp','IADD','HDATS','IDATS','HDF','HDD','HCW','HLNA']]
    Predictions4['model_id'] = model_id
    Predictions4['model_execution_id'] = model_execution_id
    Predictions4['execution_date'] = datetime.now().strftime("%Y-%m-%d")
    Predictions4['prediction_date'] = end_date 
    Predictions4['skin_code'] = brand_name

    # temporary buffer
    buff = StringIO()
    # convert data frame to csv type
    for row in Predictions4.values.tolist():
        buff.write('{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n'.format(*row))
    #Storing results into vertica table 
    vertica_connection = vertica_python.connect(**conn_info)
    with vertica_connection.cursor() as cursor:
        cursor.execute(schema_set_query, {'schema_name': analytics_schema})
        cursor.copy(query_config['STORING_RESULTS']['RG_EXECUTE'], buff.getvalue())
        vertica_connection.commit()
    vertica_connection.close()

    #Saving Results in csv format
    Predictions = Predictions[['Customer_Id','Pred']]
    Predictions.columns = ['Customer_Id','Predicted_Risk']
    Predictions['Predicted_Risk'] = np.where(Predictions.Predicted_Risk ==0,"No Risk",np.where(Predictions.Predicted_Risk ==1,"Low Risk",np.where(Predictions.Predicted_Risk==2,"Medium Risk","High Risk")))
    Risk_Count = int(Predictions[Predictions.Predicted_Risk=="High Risk"].shape[0])
    Predictions['Brand_Name'] = brand_name
    Predictions['Prediction_Date'] = end_date
    Predictions_File = Predictions[Predictions.Predicted_Risk != "No Risk"]
    Predictions_File = pd.merge(Predictions_File,Harm_Markers_data, left_on='Customer_Id', right_on='customer_id', how='left')
    Predictions_File.drop(['customer_id'],axis=1,inplace=True)
    Predictions_File = Predictions_File[['Customer_Id','Predicted_Risk','Brand_Name','Prediction_Date','IADD', 'HDATS', 'IDATS', 'HDF', 'HDD', 'HCW', 'HLNA']]
    
    #Connecting to the DB
    vertica_connection = vertica_python.connect(**conn_info)
    vsql_cur = vertica_connection.cursor()
    # Executing the statement
    vsql_cur.execute(schema_set_query, {'schema_name': analytics_schema})
    vsql_cur.execute(query_config['RG_HARM_MARKERS'])
    #Converting into DataFrame formate
    Markers = pd.DataFrame(vsql_cur.fetchall())
    Markers.columns = [c.name for c in vsql_cur.description]
    vertica_connection.close()

    #Listing Markers data
    M_ID = list(Markers.Marker_Id)
    M_Name = list(Markers.Marker_Name)
    #Inserting record in the lists.
    M_ID.insert(0,'')
    M_ID.insert(1,'Marker_Id')
    M_Name.insert(0,'')
    M_Name.insert(1,'Marker_Name')
    #Adding All markers and their description to the prediction result file
    for i,j in zip(M_ID, M_Name):
        Predictions_File.loc[Predictions_File.shape[0]] = [i,j,'','','','','','','','','']
    #Saving prediction result file in csv formate
    Predictions_File.to_csv(r_dirpath+"Prediction.csv", index=False)
    
    #Converting into json format
    metrics_dict = {"risk_count":int(Risk_Count),"brand_name": str(brand_name),"prediction_date": str(end_date)}
    metrics_dict =json.dumps(metrics_dict)
    #Updating model_execution table
    mysql_execute(query_config['MODEL_EXECUTION_UPDATE_QUERY']['RG'],(model_id,metrics_dict,"Done",int(Test_X.shape[0]),r_dirpath,model_execution_id))

    #Update 'Result Prediction completed'
    mysql_execute(query_config['UPDATE_PROCESS_LOG'],(1,current_date_time(),"Result Prediction",process_id))
    #Updating Process Table
    mysql_execute(query_config['UPDATE_PROCESS'],(3,'Done',current_date_time(),process_id))

    #Updating process_type for back-end process
    if process_mode =="sftp":
        mysql_execute(query_config['UPDATE_PROCESS_TYPE_IN_PROCESS'],("Execute",current_date_time(),process_id))

    #Updating Notification Model Execution Status :
    mysql_execute(query_config['UPDATE_NOTIFICATION'],(current_date_time(),"Done",0,1,0,0,process_id))
    logger.info("Model Execution is Completed")

    if process_mode =="sftp":        
        # inserting record into the notification table with 16 notification_id
        mysql_execute(query_config['INSERT_NOTIFICAION'],(current_date_time(),'Model Execution completed',0,16,process_id,0,0,current_date_time()))        

    #Removing uploaded files
    Root = acc_home+configs['home']['client_home']
    Dirname = "{}/{}/{}".format(client_id,user_id,process_id)
    Dirpath = os.path.join(Root,Dirname)
    #Changing directory
    os.chdir(Dirpath)
    arr = os.listdir(Dirpath)
    #Adding script for finding fils
    if process_mode.lower() != "sftp":
        for f in arr:
            if not f.endswith(".log") and not f.endswith(".dat"):
                try:
                    os.remove(f)
                except:
                    logger.info(str(f)+ ' is not able to remove')
        logger.info("All Uploaded Files are Removed")
    else:
        for f in arr:
            try:
                PATH = os.path.join(Dirpath,f) 
                # removing directory 
                shutil.rmtree(PATH)
            except:
                logger.info(str(f)+ ' is not able to remove')
                
    #Calling dashboard_enrichment script
    try:
        if process_mode != 'sftp':
            Root = acc_home
            Dirname = "python_scripts"
            Dirpath = os.path.join(Root,Dirname)
            os.chdir(Dirpath)
            from segment_enrichment import Segment_enrichment
            Segment_enrichment(process_id,log_file)
            logger.info("Executed segment enrichment script")
            from dashboard_data_enrichment import dashboard_enrichment
            from data_validation import validation
            if process_id == upload_id:
                dashboard_enrichment(process_id,client_id,log_file)
                logger.info("Executed dashboard enrichment script")
                logger.info("dw data validation started")
                validation(client_id, user_id, process_id, process_id, process_id,model_type,False,True)
                logger.info('dw data validation completed.')
    except:
        mysql_execute(query_config['UPDATE_NOTIFICATION'],(current_date_time(),"Incremental data load is failed.",0,15,0,0,process_id))
        sys.exit()

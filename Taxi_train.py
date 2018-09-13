import numpy as np
import pandas as pd
import os
from Taxi_models import *
import tensorflow as tf
from day_of_week import vectorized_dayofweek
from sklearn.preprocessing import StandardScaler
import time

from feature_utils import *
<<<<<<< HEAD
from clean_data import clean_dataset
=======
from clean_data import clean_dataset,get_clean_data_path
from Genetic_algorithm import genetic_solution
from Genetic_utils import *
>>>>>>> af08be6... Added genetic algo to taxi_train

"""
Outline:
Need more features!
Weekend/weekday marker
Holiday marker
Split time into 24 classes, could further subdivide the time
Location could also be split into classes
could factor in daylight savings time
or sunrise/sunset.
More accurate way of measuring distance?
could add euclidean distance
Could estimate the number of blocks?
Time traveled should be highly dependant on 

Training:
Probably can just use 10m and higher validation. maybe validate at 1m or near

Next steps:
Fit_generator
Save cleaned training set!

Try scaling the classes as well? see how that effects the outcome.
Try SGD with nesterov
"""

def submit_answers(test_df,test_y_predictions):
    submission = pd.DataFrame(
        {'key': test_df.key,'fare_amount': np.squeeze(test_y_predictions)},
        columns = ['key','fare_amount'])
    submission.to_csv('submission.csv',index = False)

def load_model(taxi_input,L2,learning_rate,model_path,regions,clusters,routes):
    from keras.utils.generic_utils import get_custom_objects
    from keras.models import Model, load_model
    from keras.optimizers import Adam, SGD
    from keras.utils import multi_gpu_model
    opt = Adam(lr=learning_rate,beta_1=0.9,beta_2=0.999,decay=0)
    with tf.device("/cpu:0"):
        #model = load_model(model_path,custom_objects={'losses':losses,'value_mse_loss':value_mse_loss,'policy_log_loss':policy_log_loss})
        #if model_path == None:
        model = taxi_model_V5(taxi_input,L2,regions,clusters,routes)
        #model = taxi_model_V21(taxi_input,L2)
        #else:
        #    model = load_model(model_path)
        model.compile(optimizer=opt,loss='mean_absolute_error')
        model.summary()
    gpu_model = multi_gpu_model(model,2)
    gpu_model.compile(optimizer=opt,loss='logcosh')
    return gpu_model

#Implement check pointing. Learningrate step function. Fit_generator
def train_model(model,X,Y,num_epochs,num_batches,validation,verbosity):
    from keras.models import Model
    history = model.fit(X,Y,epochs=num_epochs,batch_size=num_batches,validation_split=validation,verbose=verbosity)

def train_with_checkpoint(model,X,Y,num_epochs,num_batches,validation,verbosity,callbacks_list):
    from keras.models import Model
    history = model.fit(X,Y,epochs=num_epochs,batch_size=num_batches,validation_split=validation,callbacks=callbacks_list,verbose=verbosity)

def predict_batch(model,X):
    return model.predict_on_batch(X)

def save_model(model,model_path):
    model.save(model_path)

def return_checkpoints(weight_path,verbosity):
    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(weight_path,monitor='val_loss',verbose=verbosity,save_best_only=True,mode='min')
    callbacks_list = [checkpoint]
    return callbacks_list

#Normalize df
def normalize_mean(df):
    normalized_df=(df-df.mean())/df.std()
    return normalized_df

def normalize_minmax(df):
    normalized_df=(df-df.min())/(df.max()-df.min())
    return normalized_df

def return_training_set(decimals,num_rows,dataset):
    #for reproducibility 
    seed = 9
    np.random.seed(seed)
    train_df = pd.read_csv('/media/shuza/HDD_Toshiba/Taxi_NYC/train.csv',nrows=num_rows)
    cleaned_dataset = clean_dataset(train_df)
    tic = time.time()
    add_hour(cleaned_dataset)
    add_day(cleaned_dataset)
    add_perimeter_distance(cleaned_dataset)
    add_location_categories(cleaned_dataset,decimals)
    toc = time.time()
    print("Adding features took ",str((toc-tic)/60),' Minutes')
    #split dataset
    X = cleaned_dataset.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','perimeter_distance','pickup_region','dropoff_region']]#'pickup_clusters','dropoff_clusters'
    y = cleaned_dataset['fare_amount']
    data_to_norm = cleaned_dataset.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','perimeter_distance']]
    data_classes_train = cleaned_dataset.loc[:,['passenger_count','hour','day','pickup_region','dropoff_region']]#'pickup_clusters','dropoff_clusters'
    #Normalized training distance and LAT,LONG
    scaler = StandardScaler().fit(data_to_norm)
    X_train_scaled = pd.DataFrame(scaler.transform(data_to_norm), index=data_to_norm.index.values, columns=data_to_norm.columns.values)
    normalized_df_train = normalize_mean(data_to_norm)

    #concat the target training set
    X_train_mean = pd.concat([normalized_df_train,data_classes_train],axis=1)
    X_train_scalar = pd.concat([X_train_scaled,data_classes_train],axis=1)
    #model vars
    taxi_input = np.array(len(X.columns)).reshape(1,)
    regions = ((2*10**decimals)+10**(decimals-1)) * 3*10**decimals
    if dataset=='default':
        X_train = X
    elif dataset=='mean':
        X_train = X_train_mean
    else:
        X_train = X_train_scalar
    return X_train,y,taxi_input,regions

def return_test_set(decimals,dataset):
    test_df = pd.read_csv('/media/shuza/HDD_Toshiba/Taxi_NYC/test.csv')
    add_hour(test_df)
    add_day(test_df)
    add_perimeter_distance(test_df)
    add_location_categories(test_df,decimals)
    #test set
    test_X = test_df.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','perimeter_distance','pickup_region','dropoff_region']]
    data_to_norm_test = test_df.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','perimeter_distance']]
    data_classes_test = test_df.loc[:,['passenger_count','hour','day','pickup_region','dropoff_region']]
    #Normalized test distance and LAT,LONG
    X_test_scaled = pd.DataFrame(scaler.transform(data_to_norm_test), index=data_to_norm_test.index.values, columns=data_to_norm_test.columns.values)
    normalized_df_test = normalize_mean(data_to_norm_test)
    #concat target test set
    X_test_mean = pd.concat([normalized_df_test,data_classes_test],axis=1)
    X_test_scalar = pd.concat([X_test_scaled,data_classes_test],axis=1)
    if dataset=='default':
        X_test = test_X
    elif dataset=='mean':
        X_test = X_test_mean
    else:
        X_test = X_test_scalar
    return X_test

def main(decimals,num_rows,clusters,routes):
    #for reproducibility 
    seed = 9
    np.random.seed(seed)
<<<<<<< HEAD
    train_df = pd.read_csv('/media/shuza/HDD_Toshiba/Taxi_NYC/train.csv',nrows=num_rows)
    cleaned_dataset = clean_dataset(train_df)
=======
    training_data_path = get_clean_data_path()
    try:
        with open(training_data_path) as clean_train_csv:
            cleaned_dataset = pd.read_csv(clean_train_csv,nrows=num_rows)
        tic = time.time()
        add_hour(cleaned_dataset)
        #add_24_hour(cleaned_dataset)
        add_day(cleaned_dataset)
        add_perimeter_distance(cleaned_dataset)
        add_location_categories(cleaned_dataset,decimals) # 2 decimals = 200 * 300 = 60k
        #add_holidays(cleaned_dataset)
        #region_clusters = add_K_mean_regions(cleaned_dataset,clusters)
        #grid_clusters = add_K_mean_grid_routes(cleaned_dataset,routes)
        #default_clusters = add_K_mean_routes(df,routes)
        #np.savetxt(self.plot_path+str(i)+".txt", numpy_loss_history, delimiter=",")
        toc = time.time()
        print("Adding features took ",str((toc-tic)/60),' Minutes')

        #print(cleaned_dataset.isnull().sum(),'sum of nulls')
        #save Cluster centers so don't have to recalculate them

        #split dataset
        X = cleaned_dataset.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','perimeter_distance','pickup_region','dropoff_region','route_grid_clusters']]#'pickup_clusters','dropoff_clusters'
        y = cleaned_dataset['fare_amount']
        data_to_norm = cleaned_dataset.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','perimeter_distance']]
        data_classes_train = cleaned_dataset.loc[:,['passenger_count','hour','day','pickup_region','dropoff_region','route_grid_clusters']]#'pickup_clusters','dropoff_clusters'
        #Normalized training distance and LAT,LONG
        scaler = StandardScaler().fit(data_to_norm)
        X_train_scaled = pd.DataFrame(scaler.transform(data_to_norm), index=data_to_norm.index.values, columns=data_to_norm.columns.values)
        normalized_df_train = normalize_mean(data_to_norm)

        #concat the target training set
        X_train_mean = pd.concat([normalized_df_train,data_classes_train],axis=1)
        X_train_scalar = pd.concat([X_train_scaled,data_classes_train],axis=1)

        #model vars
        taxi_input = np.array(len(X.columns)).reshape(1,)
        regions = ((2*10**decimals)+10**(decimals-1)) * 3*10**decimals
        print(regions,'regions')
        L2 = 0.01
        alpha = 0.002
        learning_rate=0.002
        #Will have to adjust this to local directory
        weight_path = os.path.join(os.path.dirname(sys.argv[0]), "weights/weights_V5_best.hdf5")
        model_path = os.path.join(os.path.dirname(sys.argv[0]), "models/V5_checkpoint")
        #weight_path = '/media/shuza/HDD_Toshiba/Taxi_NYC/weights/weights_V5_best.hdf5'
        #model_path = '/media/shuza/HDD_Toshiba/Taxi_NYC/models/V5_checkpoint'
        verbosity = 1
        num_epochs = 100
        num_batches = 1024
        validation = 0.05
        #Checkpoints
        checkpoint = return_checkpoints(weight_path,verbosity)

        model = load_model(taxi_input,L2,learning_rate,model_path,regions,clusters,routes)
        train_with_checkpoint(model,X_train_scalar,y,num_epochs,num_batches,validation,verbosity,checkpoint)
        #Train without checkpoints
        #train(model,X_train_mean,y,num_epochs,num_batches,validation,verbosity)
        save_model(model,model_path)

        #Will have to adjust this to local directory

        test_path = os.path.join(os.path.dirname(sys.argv[0]), "test.csv")
        test_df = pd.read_csv(test_path)
        #test_df = pd.read_csv('/media/shuza/HDD_Toshiba/Taxi_NYC/test.csv')
        add_hour(test_df)
        #add_24_hour(test_df)
        add_day(test_df)
        add_perimeter_distance(test_df)
        add_location_categories(test_df,decimals) # 2 decimals = 200 * 300 = 60k
        #add_holidays(test_df)
        #add_K_mean_regions(test_df,clusters)
        #add_K_mean_grid_routes(test_df,routes)


        #test set
        test_X = test_df.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','perimeter_distance','pickup_region','dropoff_region','route_grid_clusters']]#'pickup_clusters','dropoff_clusters',
        data_to_norm_test = test_df.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','perimeter_distance']]
        data_classes_test = test_df.loc[:,['passenger_count','hour','day','pickup_region','dropoff_region','route_grid_clusters']]#'pickup_clusters','dropoff_clusters',
        #Normalized test distance and LAT,LONG
        X_test_scaled = pd.DataFrame(scaler.transform(data_to_norm_test), index=data_to_norm_test.index.values, columns=data_to_norm_test.columns.values)
        normalized_df_test = normalize_mean(data_to_norm_test)
        #concat target test set
        X_test_mean = pd.concat([normalized_df_test,data_classes_test],axis=1)
        X_test_scalar = pd.concat([X_test_scaled,data_classes_test],axis=1)

        #swap out the Y value for whichever type of dataset you want
        test_y_predictions = predict_batch(model,X_test_scalar)
        #create answer csv file
        submit_answers(test_df,test_y_predictions)
    except IOError as e:
        create_clean_dataset()

def select_genetic_model(panda_dictionary):
    from keras.optimizers import Adam, SGD
    from keras.layers import Dense
    #for reproducibility 
    seed = 9
    np.random.seed(seed)
    training_data_path = get_clean_data_path()
    with open(training_data_path) as clean_train_csv:
        cleaned_dataset = pd.read_csv(clean_train_csv,nrows=panda_dictionary['num_rows'])
>>>>>>> af08be6... Added genetic algo to taxi_train
    tic = time.time()
    add_hour(cleaned_dataset)
    #add_24_hour(cleaned_dataset)
    add_day(cleaned_dataset)
    add_perimeter_distance(cleaned_dataset)
<<<<<<< HEAD
    add_location_categories(cleaned_dataset,decimals) # 2 decimals = 200 * 300 = 60k
    #add_holidays(cleaned_dataset)
    #region_clusters = add_K_mean_regions(cleaned_dataset,clusters)
    #grid_clusters = add_K_mean_grid_routes(cleaned_dataset,routes)
    #default_clusters = add_K_mean_routes(df,routes)
    #np.savetxt(self.plot_path+str(i)+".txt", numpy_loss_history, delimiter=",")
    toc = time.time()
    print("Adding features took ",str((toc-tic)/60),' Minutes')

    #print(cleaned_dataset.isnull().sum(),'sum of nulls')
    #save Cluster centers so don't have to recalculate them

    #split dataset
    X = cleaned_dataset.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','perimeter_distance','pickup_region','dropoff_region']]#'pickup_clusters','dropoff_clusters'
    y = cleaned_dataset['fare_amount']
    data_to_norm = cleaned_dataset.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','perimeter_distance']]
    data_classes_train = cleaned_dataset.loc[:,['passenger_count','hour','day','pickup_region','dropoff_region']]#'pickup_clusters','dropoff_clusters'
=======
    add_location_categories(cleaned_dataset,panda_dictionary['decimals']) # 2 decimals = 200 * 300 = 60k
    toc = time.time()
    print("Adding features took ",str((toc-tic)/60),' Minutes')
    #split dataset
    X = cleaned_dataset.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','perimeter_distance','pickup_region','dropoff_region','route_grid_clusters']]#'pickup_clusters','dropoff_clusters'
    y = cleaned_dataset['fare_amount']
    data_to_norm = cleaned_dataset.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','perimeter_distance']]
    data_classes_train = cleaned_dataset.loc[:,['passenger_count','hour','day','pickup_region','dropoff_region','route_grid_clusters']]#'pickup_clusters','dropoff_clusters'
>>>>>>> af08be6... Added genetic algo to taxi_train
    #Normalized training distance and LAT,LONG
    scaler = StandardScaler().fit(data_to_norm)
    X_train_scaled = pd.DataFrame(scaler.transform(data_to_norm), index=data_to_norm.index.values, columns=data_to_norm.columns.values)
    normalized_df_train = normalize_mean(data_to_norm)

    #concat the target training set
    X_train_mean = pd.concat([normalized_df_train,data_classes_train],axis=1)
    X_train_scalar = pd.concat([X_train_scaled,data_classes_train],axis=1)

    #model vars
<<<<<<< HEAD
    taxi_input = np.array(len(X.columns)).reshape(1,)
    regions = ((2*10**decimals)+10**(decimals-1)) * 3*10**decimals
    print(regions,'regions')
    L2 = 0.01
    alpha = 0.002
    learning_rate=0.002
    #Will have to adjust this to local directory
    weight_path = '/media/shuza/HDD_Toshiba/Taxi_NYC/weights/weights_V5_best.hdf5'
    model_path = '/media/shuza/HDD_Toshiba/Taxi_NYC/Models/V5_checkpoint'
    verbosity = 1
    num_epochs = 100
    num_batches = 1024
    validation = 0.05
    #Checkpoints
    checkpoint = return_checkpoints(weight_path,verbosity)

    model = load_model(taxi_input,L2,learning_rate,model_path,regions,clusters,routes)
    train_with_checkpoint(model,X_train_scalar,y,num_epochs,num_batches,validation,verbosity,checkpoint)
    #Train without checkpoints
    #train(model,X_train_mean,y,num_epochs,num_batches,validation,verbosity)
    save_model(model,model_path)

    #Will have to adjust this to local directory
    test_df = pd.read_csv('/media/shuza/HDD_Toshiba/Taxi_NYC/test.csv')
    add_hour(test_df)
    #add_24_hour(test_df)
    add_day(test_df)
    add_perimeter_distance(test_df)
    add_location_categories(test_df,decimals) # 2 decimals = 200 * 300 = 60k
    #add_holidays(test_df)
    #add_K_mean_regions(test_df,clusters)
    #add_K_mean_grid_routes(test_df,routes)


    #test set
    test_X = test_df.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','perimeter_distance','pickup_region','dropoff_region','route_grid_clusters']]#'pickup_clusters','dropoff_clusters',
    data_to_norm_test = test_df.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','perimeter_distance']]
    data_classes_test = test_df.loc[:,['passenger_count','hour','day','pickup_region','dropoff_region','route_grid_clusters']]#'pickup_clusters','dropoff_clusters',
    #Normalized test distance and LAT,LONG
    X_test_scaled = pd.DataFrame(scaler.transform(data_to_norm_test), index=data_to_norm_test.index.values, columns=data_to_norm_test.columns.values)
    normalized_df_test = normalize_mean(data_to_norm_test)
    #concat target test set
    X_test_mean = pd.concat([normalized_df_test,data_classes_test],axis=1)
    X_test_scalar = pd.concat([X_test_scaled,data_classes_test],axis=1)

    #swap out the Y value for whichever type of dataset you want
    test_y_predictions = predict_batch(model,X_test_scalar)
    #create answer csv file
    submit_answers(test_df,test_y_predictions)

#clusters = 1500
#routes = 10000
#decimals = 2
#num_rows = 10000000
#main(decimals,num_rows,clusters,routes)
=======
    regions = ((2*10**panda_dictionary['decimals'])+10**(panda_dictionary['decimals']-1)) * 3*10**panda_dictionary['decimals']
    print(regions,'regions')
    #Will have to adjust this to local directory
    weight_path = os.path.join(os.path.dirname(sys.argv[0]), "weights/weights_V5_best.hdf5")
    model_path = os.path.join(os.path.dirname(sys.argv[0]), "models/V5_checkpoint")
    #weight_path = '/media/shuza/HDD_Toshiba/Taxi_NYC/weights/weights_V5_best.hdf5'
    #model_path = '/media/shuza/HDD_Toshiba/Taxi_NYC/models/V5_checkpoint'
    verbosity = 1
    checkpoint = return_checkpoints(weight_path,verbosity)

    ### Model initialization vars ###
    learning_rates = [0.08,0.05,0.02,0.002]
    alpha = [0.01,0.001,0.05,0.02,0.002]
    L2s = []
    dropout_rates = []
    filters = []
    strides = []
    L2 = 0.01
    alpha = 0.002

    opt_adam = Adam(lr=learning_rates[2],beta_1=0.9,beta_2=0.999,decay=0)
    opt_sgd = SGD(lr=learning_rates[2],nesterov=True)
    optimizations = ['sgd+nestorov','adam']
    ###

    ### Model Attributes ###
    #input_shape = (int(X.shape[-2]),int(X.shape[-1]),)
    input_shape = np.array(len(X.columns)).reshape(1,)
    num_param_choices = 4
    hyperparams_choices = [0,1,2,3]
    num_hyperparams = 4
    #main layer
    layer_choices = np.array([0,1,2,3]) #len(layer_dictionary.keys())
    num_layer_choices = 4
    #initial layer
    initial_layer_choices = np.array([4])
    num_initial_choices = 2
    #model length
    num_total_layers = 8
    num_different_layers = 2
    layer_ranges = np.array([2,num_total_layers])
    num_modifications = 2 #Per model
    layers_to_modify = list(range(num_total_layers))
    params_to_modify = [0,1] #-1 for no impact
    mod_layers = True
    mod_params = False
    mod_hyperparams = True
    multigpu = False
    initialize = True
    N = 5 #number of models
    #For initializing model arch
    num_main_layers = N*(layer_ranges[1]-layer_ranges[0])
    num_initial_layers = N*num_initial_choices
    model_dictionary = {'input_shape':input_shape,'num_param_choices':num_param_choices,'num_hyperparams':num_hyperparams,
                    'layer_choices':layer_choices,'num_total_layers':num_total_layers,
                        'layer_ranges':layer_ranges,'num_initial_layers':num_initial_layers,
                        'num_modifications':num_modifications,'num_different_layers':num_different_layers,
                        'initial_layer_choices':initial_layer_choices,'num_main_layers':num_main_layers,
                    'N':N,'modifications_per_model':N*num_modifications,'optimizations':optimizations,
                        'learning_rates':learning_rates,'layer_keys':np.repeat(np.arange(N),num_modifications),
                        'hyperparams_choices':hyperparams_choices,'initialize':initialize,
                        'layers_to_modify':layers_to_modify,'params_to_modify':params_to_modify,
                        'mod_layers':mod_layers,'mod_params':mod_params,'mod_hyperparams':mod_hyperparams,'multigpu':multigpu
                    }
    ### END ###

    ### Training dictionary params ###
    learning_rate = 0.02
    val_split = 0.15
    num_epochs = 5
    num_batches = 1024
    model_folder = '/Users/Shuza/Code/Models'
    callbacks_list = None
    iterations = 25
    training_dictionary = {'learning_rate':learning_rate,'val_split':val_split,'num_epochs':num_epochs,
                        'model_folder':model_folder,'callbacks':callbacks_list,'iterations':iterations,
                        'index':np.arange(num_total_layers),'layer_keys':np.repeat(np.arange(N),num_modifications),'num_batches':num_batches}
    ### End ###
    ### Layer and Dimension Dictionaries ###
    layer_dictionary = {
            0:{0:make_dense_relu,1:make_dense_tanh,2:make_empty_layer,4:make_conv2d_layer,5:make_conv1d_layer},
            1:{0:make_dense_relu,1:make_dense_tanh,2:make_empty_layer,4:make_conv2d_layer,5:make_conv1d_layer},
            2:{0:make_dense_relu,1:make_dense_tanh,2:make_empty_layer,3:skip_layer},
            3:{0:make_dense_relu,1:make_dense_tanh,2:make_empty_layer,3:skip_layer},
            4:{0:make_dense_relu,1:make_dense_tanh,2:make_empty_layer,3:skip_layer},
            5:{0:make_dense_relu,1:make_dense_tanh,2:make_empty_layer,3:skip_layer},
            6:{0:make_dense_relu,1:make_dense_tanh,2:make_empty_layer,3:skip_layer},
            7:{0:make_dense_relu,1:make_dense_tanh,2:make_empty_layer,3:skip_layer},
            'output_layer':Dense(1,activation='relu')
            }

    dimension_dictionary = {
        2:{0:make_dense_relu,1:make_dense_tanh,2:make_empty_layer,3:skip_layer},
        3:{0:make_dense_relu,1:make_dense_tanh,2:make_empty_layer,3:skip_layer,5:make_conv1d_layer},
        4:{0:make_dense_relu,1:make_dense_tanh,2:make_empty_layer,3:skip_layer,4:make_conv2d_layer,5:make_conv1d_layer},
        5:{0:make_dense_relu,1:make_dense_tanh,2:make_empty_layer,3:skip_layer,4:make_conv2d_layer,5:make_conv1d_layer}
    }
    ### END ###
    layers,params,hyperparam = genetic_solution(X,y,model_dictionary,layer_dictionary,training_dictionary,dimension_dictionary)
    print(layers,params,hyperparam,'best model')
    #model = load_model(taxi_input,L2,learning_rate,model_path,regions,clusters,routes)
    #train_with_checkpoint(model,X_train_scalar,y,num_epochs,num_batches,validation,verbosity,checkpoint)


def create_clean_dataset():
    print("BuildingFile: 'clean_train.csv' doesn't exist. Building file for future use. Rerun program once done.")
    training_path = os.path.join(os.getcwd(), "train.csv")
    train_df = pd.read_csv(training_path, nrows=55423856)
    clean_dataset(train_df)

clusters = 1500
routes = 10000
decimals = 2
num_rows = 100000
panda_dictionary = {
    'clusters':clusters,
    'routes':routes,
    'decimals':decimals,
    'num_rows':num_rows
}
#create_clean_dataset()
select_genetic_model(panda_dictionary)
#main(decimals,num_rows,clusters,routes)
>>>>>>> af08be6... Added genetic algo to taxi_train

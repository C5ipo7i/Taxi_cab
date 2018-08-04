import numpy as np
import pandas as pd
import os
from Taxi_models import *
import tensorflow as tf
from day_of_week import vectorized_dayofweek
from sklearn.preprocessing import StandardScaler
import time

from feature_utils import *
from clean_data import clean_dataset

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

def main(decimals,num_rows,clusters,routes):
    #for reproducibility 
    seed = 9
    np.random.seed(seed)
    train_df = pd.read_csv('/media/shuza/HDD_Toshiba/Taxi_NYC/train.csv',nrows=num_rows)
    cleaned_dataset = clean_dataset(train_df)
    tic = time.time()
    add_hour(cleaned_dataset)
    #add_24_hour(cleaned_dataset)
    add_day(cleaned_dataset)
    add_perimeter_distance(cleaned_dataset)
    add_location_categories(cleaned_dataset,decimals) # 2 decimals = 200 * 300 = 60k
    #add_holidays(cleaned_dataset)
    region_clusters = add_K_mean_regions(cleaned_dataset,clusters)
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
    add_K_mean_regions(test_df,clusters)
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

clusters = 1500
routes = 10000
decimals = 2
num_rows = 10000000
main(decimals,num_rows,clusters,routes)
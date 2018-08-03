from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model
import tensorflow as tf

import numpy as np
from sklearn.preprocessing import StandardScaler

from Taxi_train import submit_answers,normalize_mean,predict_batch
from feature_utils import *
from Taxi_models import *

#For making predictions with checkpointed gpu models
decimals = 2
#load data
test_df = pd.read_csv('/media/shuza/HDD_Toshiba/Taxi_NYC/test.csv')
#add_hour(test_df)
add_24_hour(cleaned_dataset)
add_day(test_df)
add_perimeter_distance(test_df)
add_location_categories(test_df,decimals) # 2 decimals = 200 * 300 = 60k

test_X = test_df.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','hour','day','perimeter_distance','pickup_region','dropoff_region']]
data_to_norm_test = test_df.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','perimeter_distance']]
data_classes_test = test_df.loc[:,['passenger_count','hour','day','pickup_region','dropoff_region']]
#Normalized test distance and LAT,LONG
scaler = StandardScaler().fit(data_to_norm_test)
X_test_scaled = pd.DataFrame(scaler.transform(data_to_norm_test), index=data_to_norm_test.index.values, columns=data_to_norm_test.columns.values)
normalized_df_test = normalize_mean(data_to_norm_test)
#concat target test set
X_test_mean = pd.concat([normalized_df_test,data_classes_test],axis=1)
X_test_scalar = pd.concat([X_test_scaled,data_classes_test],axis=1)

#model vars
taxi_input = np.array(len(X_test_mean.columns)).reshape(1,)
L2 = 0.01
alpha = 0.002
learning_rate=0.002
regions = ((2*10**decimals)+10**(decimals-1)) * 3*10**decimals

#load checkpoint model
model_path = '/media/shuza/HDD_Toshiba/Taxi_NYC/Models/V4_checkpoint'
weight_path = '/media/shuza/HDD_Toshiba/Taxi_NYC/weights/weights_V4_best.hdf5'

#compile model with hyperparams
opt = Adam(lr=learning_rate,beta_1=0.9,beta_2=0.999,decay=0)
with tf.device("/cpu:0"):
    #model = load_model(model_path,custom_objects={'losses':losses,'value_mse_loss':value_mse_loss,'policy_log_loss':policy_log_loss})
    #if model_path == None:
    model = taxi_model_V4(taxi_input,L2,regions)
    #else:
    #    model = load_model(model_path)
    model.compile(optimizer=opt,loss='mean_absolute_error')
    model.summary()
gpu_model = multi_gpu_model(model,2)
gpu_model.compile(optimizer=opt,loss='logcosh')
gpu_model.load_weights(weight_path)

test_y_predictions = predict_batch(model,X_test_scalar)
#create answer csv file
submit_answers(test_df,test_y_predictions)
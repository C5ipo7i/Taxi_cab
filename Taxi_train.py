import numpy as np
import pandas as pd
import os
from Taxi_models import taxi_model
import tensorflow as tf
from day_of_week import vectorized_dayofweek

"""
Outline:
Weekend/weekday marker
Holiday marker
Split time into 24 classes
Location could also be split into classes
"""

def load_model(taxi_input,L2,learning_rate,model_path):
    from keras.utils.generic_utils import get_custom_objects
    from keras.models import Model, load_model
    from keras.optimizers import Adam, SGD
    from keras.utils import multi_gpu_model
    opt = Adam(lr=learning_rate,beta_1=0.9,beta_2=0.999,decay=0)
    with tf.device("/cpu:0"):
        #model = load_model(model_path,custom_objects={'losses':losses,'value_mse_loss':value_mse_loss,'policy_log_loss':policy_log_loss})
        if model_path == None:
            model = taxi_model(taxi_input,L2)
        else:
            model = load_model(model_path)
        model.compile(optimizer=opt,loss='mse')
        model.summary()
    gpu_model = multi_gpu_model(model,2)
    gpu_model.compile(optimizer=opt,loss='logcosh')
    return gpu_model

def train_model(model,X,Y,num_epochs,verbosity):
    from keras.models import Model
    history = model.fit(X,Y,epochs=num_epochs,verbose=verbosity)

def predict_batch(model,X):
    return model.predict_on_batch(X)

def save_model(model,model_path):
    model.save(model_path)

#round hour to nearest hour? will have to check about crossing into the next or previous day
#vectorize the functions
def add_hour(df):
    new_test = df.pickup_datetime.str.split().str.get(-2).str.split(':').str.get(0)
    hours = pd.to_numeric(new_test, errors='coerce')
    df['hour'] = hours

def add_day(df):
    new_test = df.pickup_datetime.str.split().str.get(0).str.split('-')
    year = pd.to_numeric(new_test.str.get(0),errors='coerce')
    month = pd.to_numeric(new_test.str.get(1),errors='coerce')
    day = pd.to_numeric(new_test.str.get(2),errors='coerce')
    print(day.values)
    final_day = vectorized_dayofweek(day.values,month.values,year.values)
    df['day'] = final_day
    print('hi')

train_df = pd.read_csv('/media/shuza/HDD_Toshiba/Taxi_NYC/train.csv',nrows=10)
add_hour(train_df)
add_day(train_df)

#split dataset
#X = train_df.drop(['fare_amount','pickup_datetime'],axis=0)
X = train_df.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']]
y = train_df['fare_amount']

#model vars
taxi_input = np.array(len(X.columns)).reshape(1,)
L2 = 0.01
alpha = 0.002
learning_rate=0.002
#Will have to adjust this to local directory
model_path = '/media/shuza/HDD_Toshiba/Taxi_NYC/Models/V1'
verbosity = 1
num_epochs = 400

model = load_model(taxi_input,L2,learning_rate,model_path)
train_model(model,X,y,num_epochs,verbosity)
save_model(model,model_path)

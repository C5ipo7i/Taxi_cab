from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense,Dropout, Activation, Lambda,Flatten,Conv1D,Conv2D,LeakyReLU
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Add
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.models import load_model, Model, Sequential
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.callbacks import Callback,LearningRateScheduler
from keras.models import load_model
from keras.layers import InputLayer
from keras.utils import multi_gpu_model
import keras.backend as K

def taxi_model(Input_shape,L2):
    X_input = Input(Input_shape)
    X = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(L2))(X_input)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn0')(X)
    X = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn1')(X)
    X = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn2')(X)
    X = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn3')(X)
    X = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn4')(X)
    X = Dense(1)(X)
    model = Model(inputs = X_input, outputs = X, name='multimodel')
    return model

#add skip connections.
def taxi_model_V2(Input_shape,L2):
    X_input = Input(Input_shape)
    X_skip = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X_input)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn0')(X_skip)
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn1')(X)
    X = Add()([X_skip,X])
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn2')(X)
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn3')(X)
    X = Add()([X_skip,X])
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn4')(X)
    X = Dense(1)(X)
    model = Model(inputs = X_input, outputs = X, name='multimodel')
    return model

#add new type of skip connections.
def taxi_model_V21(Input_shape,L2):
    X_input = Input(Input_shape)
    X_skip = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X_input)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn0')(X_skip)
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn1')(X)
    X = Add()([X_skip,X])
    X_skip_2 = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn2')(X_skip_2)
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn3')(X)
    X = Add()([X_skip,X_skip_2,X])
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn4')(X)
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn5')(X)
    X = Dense(1)(X)
    model = Model(inputs = X_input, outputs = X, name='multimodel')
    return model

def split_data(tensor):
    return tensor[:,:5]

def split_class(tensor):
    return tensor[:,5:]

def split_passengers(tensor):
    return tensor[:,0]
def split_hour(tensor):
    return tensor[:,1]
def split_day(tensor):
    return tensor[:,2]
def split_pickup(tensor):
    return tensor[:,3]
def split_dropoff(tensor):
    return tensor[:,4]


#Think the classes are -2 and -3? recheck this
def taxi_model_V3(Input_shape,L2):
    X_input = Input(Input_shape)

    data = Lambda(split_data,output_shape=(5,))(X_input)
    classes = Lambda(split_class,output_shape=(3,))(X_input)
    passengers = Lambda(split_passengers,output_shape=(1,))(classes)
    hour = Lambda(split_hour,output_shape=(1,))(classes)
    day = Lambda(split_day,output_shape=(1,))(classes)

    print(data.shape,classes.shape,passengers.shape,hour.shape,day.shape,'inputs')
    C1 = Embedding(7,32,)(passengers)
    C2 = Embedding(25,512,)(hour)
    C3 = Embedding(8,256,)(day)
    #data = Reshape((1,5))(data)

    X_start_2 = Concatenate(axis=-1)([C1,C2,C3])
    #X_class = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(L2))(X_start_2)
    #X_data = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(L2))(data)
    X_start_1 = Reshape((800,))(X_start_2)
    print(X_start_1.shape,data.shape,'first concate')
    #print(X_class.shape,data.shape,'first concate')
    #X_start = Concatenate(axis=-1)([X_class,X_data])
    X_start = Concatenate(axis=-1)([X_start_1,data])
    print(X_start.shape,'after concate')

    X_skip = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X_start)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn0')(X_skip)
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn1')(X)
    print(X_skip.shape,X.shape,'prior to add')
    X = Add()([X_skip,X])
    X_skip_2 = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn2')(X_skip_2)
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn3')(X)
    print(X_skip_2.shape,X.shape,'prior to add')
    X = Add()([X_skip_2,X])
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn4')(X)
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn5')(X)
    X = Dense(1)(X)
    model = Model(inputs = X_input, outputs = X, name='multimodel')
    return model

def taxi_model_V4(Input_shape,L2,regions):
    X_input = Input(Input_shape)

    data = Lambda(split_data,output_shape=(5,))(X_input)
    classes = Lambda(split_class,output_shape=(3,))(X_input)
    print(classes.shape,'classes')
    passengers = Lambda(split_passengers,output_shape=(1,))(classes)
    hour = Lambda(split_hour,output_shape=(1,))(classes)
    day = Lambda(split_day,output_shape=(1,))(classes)
    pickup = Lambda(split_pickup,output_shape=(1,))(classes)
    dropoff = Lambda(split_dropoff,output_shape=(1,))(classes)

    print(data.shape,classes.shape,passengers.shape,hour.shape,day.shape,pickup.shape,dropoff.shape,'inputs')

    location = Embedding(regions,64)

    C1 = Embedding(7,32,)(passengers)
    C2 = Embedding(169,128,)(hour)
    C3 = Embedding(8,128,)(day)
    C4 = location(pickup)
    C5 = location(dropoff)
    #data = Reshape((1,5))(data)

    X_start_2 = Concatenate(axis=-1)([C1,C2,C3,C4,C5])
    print(X_start_2.shape,'first concate')
    #X_class = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(L2))(X_start_2)
    #X_data = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(L2))(data)
    X_start_1 = Reshape((416,))(X_start_2)
    print(X_start_1.shape,data.shape,'first concate')
    #print(X_class.shape,data.shape,'first concate')
    #X_start = Concatenate(axis=-1)([X_class,X_data])
    X_start = Concatenate(axis=-1)([X_start_1,data])
    #print(X_start.shape,'after concate')

    X_skip = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X_start)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn0')(X_skip)
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn1')(X)
    print(X_skip.shape,X.shape,'prior to add')
    X = Add()([X_skip,X])
    X_skip_2 = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn2')(X_skip_2)
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn3')(X)
    print(X_skip_2.shape,X.shape,'prior to add')
    X = Add()([X_skip,X_skip_2,X])
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    #X = Dropout(0.8)(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn4')(X)
    X = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(L2))(X)
    #X = Dropout(0.8)(X)
    X = BatchNormalization(axis=1, epsilon=0.00001, name='bn5')(X)
    X = Dense(1)(X)
    model = Model(inputs = X_input, outputs = X, name='multimodel')
    return model
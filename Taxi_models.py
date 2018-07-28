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

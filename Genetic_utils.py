
from keras.layers import Input,Reshape,Dense,Activation, Lambda,Flatten,Conv1D,Conv2D,LeakyReLU,BatchNormalization,Add

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

#Layers
def make_dense_relu(X,params):
    X = Dense(64,activation='relu')(X)
    return X

def make_dense_tanh(X,params):
    X = Dense(64,activation='tanh')(X)
    return X

def make_empty_layer(X,params):
    return X

def skip_layer(X,params):
    # defining name basis
    #bn_name_base = 'bn' + str(stage) + '_branch'
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    #print(X_shortcut.shape,'X_shortcut.shape')
    skip_shape = int(X_shortcut.shape[-1])
    # First component of main path
    X = Dense(64)(X)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu')(X)
    # Second component of main path (≈3 lines)
    X = Dense(skip_shape)(X)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    #print(X.shape,'final shape')
    return X

#f = (1,3),stride = (1,3) for ex
def make_conv2d_layer(X,params):
    #padding_dictionary = {0:'valid',1:'same'}
    print(X.shape,'2d conv input shape')
    print(len(list(X.shape)),'len')
    print(params,'params 2d')
    if len(list(X.shape)) == 3:
        X = Reshape((int(X.shape[-2]),int(X.shape[-1]),1))(X)
    elif len(list(X.shape)) == 2:
        X = Reshape((int(X.shape[-1]),1,1))(X)
    f,s,p = params
    #print(f,s,'fs')
    #check dimensions of X, reshape to (?,1) if necessary
    #X = Conv2D(64, f, strides = s, padding = padding_dictionary[p])(X)
    X = Conv2D(64, f, strides = s, padding = p)(X)
    return X

#f = 3,stride = 3 for ex
def make_conv1d_layer(X,params):
    #padding_dictionary = {0:'valid',1:'same'}
    print(params,'params 1d')
    print(len(list(X.shape)),'len')
    if len(list(X.shape)) == 2:
        X = Reshape((int(X.shape[-1]),1))(X)
    f,s,p = params
    #print(f,s,'fs')
    #check dimensions of X, reshape to (?,1) if necessary
    #X = Conv1D(64, int(f), strides = int(s), padding = padding_dictionary[p])(X)
    X = Conv1D(64, int(f), strides = int(s), padding = p)(X)
    return X

def max_pool2d(X,params):
    print(params,'params 1d')
    print(len(list(X.shape)),'len')
    if len(list(X.shape)) == 2:
        X = Reshape((int(X.shape[-1]),1))(X)
    f,s,p = params[0]
    print(f,s,'fs')
    #check dimensions of X, reshape to (?,1) if necessary
    X = MaxPooling2D(pool_size = f, strides = s, padding = p)(X)
    return X

def max_pool1d(X,params):
    print(params,'params 1d')
    print(len(list(X.shape)),'len')
    if len(list(shape)) == 2:
        X = Reshape((int(shape[-1]),1))(X)
    f,s,p = params[0]
    print(f,s,'fs')
    #check dimensions of X, reshape to (?,1) if necessary
    X = MaxPooling1D(pool_size = f, strides = s, padding = p)(X)
    return X
import numpy as np
from multiprocessing import Pool
from keras.layers import Input,Dense,Activation,Lambda,Flatten,Conv1D,Conv2D,LeakyReLU
from keras.models import Model

from Taxi_train import return_training_set,return_test_set
from worker_class import Worker
from Utils import softmax

"""
Genetic Algorithm

randomly initialize parameters
train series of models
select best performing models -> randomly select 25% of the model to change.
run training. Repeat.
Include the best model in the training set as well. To have a base case
Should be able to modify how much you change the model. 
Also potentially reduce the randomness over time? Initially spread out over lots of different models. 
Then narrow down over time. Could retain some stragglers for variety
"""

#Layers
def make_dense_relu(X):
    X = Dense(64,activation='relu')(X)
    return X

def make_dense_tanh(X):
    X = Dense(64,activation='tanh')(X)
    return X

def skip_layer(X):
    # defining name basis
    #bn_name_base = 'bn' + str(stage) + '_branch'
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Dense(64)(X)
    X = BatchNormalization(axis = -1, name = bn_name_base + '1a')(X)
    X = Activation('relu')(X)
    # Second component of main path (≈3 lines)
    X = Dense(64)(X)
    X = BatchNormalization(axis = -1, name = bn_name_base + '1b')(X)
    X = Activation('relu')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    return X

def random_layer_initialization(num_layers,num_choices):
    initial_choice = np.random.rand(num_choices,num_layers)
    print(initial_choice.shape,'initial')
    softmax_choice = softmax(initial_choice)
    print(softmax_choice)
    indexes = np.argmax(softmax_choice,axis=0)
    print(indexes,'index')
    return indexes

#Conv layer = filter sizes (F), stride (s), kernel size (k)
#Lstm layer = ~
#Dense = Activation
#Skip connections = Activation(s)
def random_layer_hyperparams(num_layers,num_choices):
    initial_choice = np.random.rand(num_choices,num_layers)
    print(initial_choice.shape,'initial')
    softmax_choice = softmax(initial_choice)
    print(softmax_choice)
    indexes = np.argmax(softmax_choice,axis=0)
    print(indexes,'index')
    return indexes

#list of hyperparams
#L2 (0.01)
#alpha = 0.002 (for leaky relu)
#learning_rate = 0.02~
def random_hyperparams(num_params):
    pass

def make_model(input_shape,layers,layer_params,output_layer,model_dictionary):
    layer_1,layer_2,layer_3,layer_4 = layers
    param_1,param_2,param_3,param_4 = layer_params
    #print(model_dictionary[layer_1],'dict')
    X_input = Input(input_shape)
    X = model_dictionary[layer_1](X_input)
    X = model_dictionary[layer_2](X)
    X = model_dictionary[layer_3](X)
    X = model_dictionary[layer_4](X)
    X = output_layer(X)
    model = Model(inputs = X_input, outputs = X)
    return model

def train_models():
    pass

def permute_model(self):
    layers = self.get_layers()
    num_layers = len(layers)
    chosen_layer = np.random.choice(layers,1)
    new_layer = self.new_random_layer()

    self.model.summary()
    pass

def return_new_layer(num_choices):
    return np.random.rand(num_choices,1)

def return_layer_config(layer):
    conf = test_layers[random_choice].get_config()
    layer_activation = conf['activation']
    layer_name = conf['name']
    layer_type = layer_name.split('_')[0]

def return_model_configs(num_layers,num_choices):
    layers = random_layer_initialization(num_layers,num_choices)
    layer_params = random_layer_hyperparams(num_layers,num_choices)
    return layers,layer_params

#could do n number of rounds of random layer initialization first
#and then select the top n models from those rounds, and then start permuting them
def main(dataset,num_rows,decimals,num_layers,num_choices,input_shape,layer_dictionary):
    #initial vars
    #p = Pool(processes=2)
    model_dictionary = {}
    index = np.arange(num_layers)
    validation = 0.05
    workers = []
    losses = []
    output_layer = Dense(1,activation='relu')
    num_epochs = 10
    model_path = '/media/shuza/HDD_Toshiba/Taxi_NYC/Models/GA'
    callbacks_list = None
    #get training set and params
    X,y,taxi_input,regions = return_training_set(decimals,num_rows,dataset)
    #Get model
    layers = random_layer_initialization(num_layers,num_choices)
    layer_params = random_layer_hyperparams(num_layers,num_choices)
    model_dictionary[0] = (layers,layer_params)
    print(layers,layer_params,'layer data')
    model_1 = make_model(input_shape,layers,layer_params,output_layer,layer_dictionary)
    model_1.summary()
    #train models
    worker_1 = Worker(model,num_epochs,validation,model_path,callbacks_list=callbacks_list)
    worker_2 = Worker(model,num_epochs,validation,model_path,callbacks_list=callbacks_list)
    workers.append(worker_1)
    workers.append(worker_2)
    #get losses, use multiprocessing pool
    for worker in workers:
        worker.train(X,y)
    for worker in workers:
        losses.append(worker.loss)
    #select best loss
    best_computed_loss = min(losses)
    #select model
    #best_model_index = np.where(losses==best_computed_loss) #use argmin
    best_model_index = np.argmin(losses)
    #save model architecture, might be a better way
    
    #permute best model. Randomly modify a layer and remake the workers
    current_best_model_arch_test = workers[best_model_index].get_layers
    test_layers = workers[best_model_index].model.layers
    random_choice = np.random.choice(index,1)
    new_layer = return_new_layer(num_choices)
    #Grab layers from dictionary and replace the value
    #create n new model variations
    
    #retrain. Repeat X number of steps

#model_dictionary = {0:skip_layer,1:lstm_layer,2:dense_relu_layer,3:dense_tanh_layer,4:convolutional_block}

layer_dictionary = {0:make_dense_relu,1:make_dense_tanh}
dataset = 'scaled'
num_rows = 100
decimals = 1
input_shape = (1,)    
num_choices = 2
num_layers = 4
main(dataset,num_rows,decimals,num_layers,num_choices,input_shape,layer_dictionary)


#print(layer_type)
#print(test_layers[1])
#print(test_layers[1].get_config())
#method_list_conv = [func for func in dir(conf) if callable(getattr(conf, func))]
#method_list = [func for func in dir(test_layers[1]) if callable(getattr(test_layers[1], func))]
#print(method_list_conv)
#print(method_list)

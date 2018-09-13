import numpy as np
<<<<<<< HEAD
from multiprocessing import Pool
from keras.layers import Input,Dense,Activation,Lambda,Flatten,Conv1D,Conv2D,LeakyReLU
from keras.models import Model

from Taxi_train import return_training_set,return_test_set
from worker_class import Worker
from Utils import softmax
=======
import copy
import json
import random
#import tensorflow as tf
from multiprocessing import Process,Queue,Pool

from keras.optimizers import Adam, SGD
from keras.layers import Input,Reshape,Dense,Activation, Lambda,Flatten,Conv1D,Conv2D,LeakyReLU,BatchNormalization,Add
from keras.models import Model 
import keras.backend as K
from keras.utils import multi_gpu_model

from worker_class import Worker
>>>>>>> af08be6... Added genetic algo to taxi_train

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
<<<<<<< HEAD
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
=======

Instantiate n number of models, save model layer inputs (to instantiate a model) by model number
train models, retrive loss. If no best loss yet, save best model arch as 'Best' key. Otherwise compare
new best loss with previous best loss.
take best architecture and permute it n times, saving all variations (overwriting) in model dictionary.
repeat

Blueprint:
random layers return (N,choices,layers)
random params return (N,choices,params)
multi processing pool - create N models, and seed the dictionary with all archs
multi processing pool - train N models, record N losses
Check new lowest loss with previous best loss
update best model arch if necessary
create N new random layers (N,choices,layers)
create N new random params (N,choices,params)
repeat

possible improvements:
save best and 2nd best model archs
Or best and a random arch
or best + top 5 etc.
change more than 1 layer at a time
Make certain layers only available after 1st layer?

TODO:
include parameter updates not just layer switches.

1.Keep track of which model params you have tried. Store the loss in a dictionary and only search new ones
iterate through every possible version. stringify the inputs to make a key?
2.use gradient descent to choose the direction of the next update
3.modularize the functions so that its easy to pick which thing you are varying:
For ex. be able to pass the model layers through and only vary hyperparams (learning rate)
or keep hyperparams constant and very model layers.

4.implement checkpointing

5.Can select for specific end and start layers by just making sure the initial/end layer is chosen from a different
range. So eg. first layer may only have 3 choices and layer 2+ has 5 choices. and last layer has 2 choices.

Model dictionary:
Layer data
hyperparam data
layer para data
optimization

Hyperparam selection:
Each layer will take 2 inputs (but not necessarily use them all), filter,stride (could include num neurons)
we need to vary each?

Hyperparam dictionary:

Param selection:
learning_rate - (also implement step function)
optimizations - Adam,SGD

Revisit param selection based on logs

For model2.0
train model2.0 policy on the best loss model arch?
"""

def make_data_set():
    #from sklearn.datasets.samples_generator import make_blobs
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=5,n_targets=1,noise=0.01,random_state=0)
    #print(X.shape)
    #print(y)

    # Data reshape for convolution
    print(X.shape,'initial data shape')
    X = X.reshape(X.shape[-2],X.shape[-1],1)
    print(X.shape,'post data shape')
    return X,y

def merge_dicts(a,b):
    z = a.copy()
    z.update(b)
    return z

def random_layer_initialization(model_dictionary):
    #initial_choice = np.random.rand(N,num_choices,num_layers)
    #print(initial_choice.shape,'initial layers')
    #softmax_choice = softmax(initial_choice)
    #print(softmax_choice)
    #indexes = np.argmax(softmax_choice,axis=0)
    #print(indexes,'index layers')
    #return indexes
    gen_layers = np.random.choice(model_dictionary['layer_choices'],model_dictionary['num_main_layers']).reshape(model_dictionary['N'],(model_dictionary['layer_ranges'][1]-model_dictionary['layer_ranges'][0]))
    init_layers = np.random.choice(model_dictionary['initial_layer_choices'],model_dictionary['num_initial_layers']).reshape(model_dictionary['N'],model_dictionary['layer_ranges'][0])
    return np.hstack((init_layers,gen_layers))
    #return np.random.randint(0,model_dictionary['num_layer_choices'],(model_dictionary['N'],model_dictionary['num_layers']))
>>>>>>> af08be6... Added genetic algo to taxi_train

#Conv layer = filter sizes (F), stride (s), kernel size (k)
#Lstm layer = ~
#Dense = Activation
#Skip connections = Activation(s)
<<<<<<< HEAD
def random_layer_hyperparams(num_layers,num_choices):
    initial_choice = np.random.rand(num_choices,num_layers)
    print(initial_choice.shape,'initial')
    softmax_choice = softmax(initial_choice)
    print(softmax_choice)
    indexes = np.argmax(softmax_choice,axis=0)
    print(indexes,'index')
    return indexes
=======
def random_layer_hyperparams(model_dictionary):
    #initial_choice = np.random.rand(N,num_choices,num_layers)
    #print(initial_choice.shape,'initial hyperparams')
    #softmax_choice = softmax(initial_choice)
    #print(softmax_choice)
    #indexes = np.argmax(softmax_choice,axis=0)
    #print(indexes,'index params')
    return np.random.randint(0,model_dictionary['num_layer_choices'],(model_dictionary['N'],model_dictionary['num_total_layers']))
>>>>>>> af08be6... Added genetic algo to taxi_train

#list of hyperparams
#L2 (0.01)
#alpha = 0.002 (for leaky relu)
#learning_rate = 0.02~
<<<<<<< HEAD
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
=======
def random_hyperparams(model_dictionary):
    #return np.random.randint(0,model_dictionary['num_hyperparams'],(model_dictionary['N']),dtype=int)
    return random.randint(0,model_dictionary['num_hyperparams'],(model_dictionary['N']))
    
def get_layer(layer_dictionary,dimension_dictionary,shape,layer):
    #print(shape,'shape')
    dimensions = len(list(shape))
    #print(dimensions,'N dimensions')
    #print(layer,'layer num')
    layer_choices = layer_dictionary[layer]
    #get total choices from the dimension given
    #take the intersection of the choices
    available_choices = dimension_dictionary[dimensions]
    choices = {x:layer_choices[x] for x in layer_choices if x in available_choices}
    #print(choices,'final choices')
    choice = int(np.random.choice(list(choices)))
    #print(type(choice))
    #if improper dimensions then remove conv2d and conv1d
    params = get_param(choice,shape)
    return choice,params

def get_param(layer,shape):
    if layer == 4: #if conv then choose which conv based on input length
        first = int(shape[-2])+1
        second = int(shape[-1])+1
        #2d
        #kernel
        print('2d conv')
        kernel_1 = int(np.random.choice(np.arange(1,first,dtype=int)))
        kernel_2 = int(np.random.choice(np.arange(1,second,dtype=int)))
        #stride
        stride_1 = int(np.random.choice(np.arange(1,first,dtype=int)))
        stride_2 = int(np.random.choice(np.arange(1,second,dtype=int)))
        #padding
        padding = np.random.choice(['same','valid'])
        params = [(kernel_1,kernel_2),(stride_1,stride_2),padding]
        #print(params,'2d params')
    elif layer == 5:
        first = int(shape[-2])+1
        second = int(shape[-1])+1
        #1d
        #kernel
        print('1d conv')
        kernel_1 = int(np.random.choice(np.arange(1,first,dtype=int)))
        #stride
        stride_1 = int(np.random.choice(np.arange(1,second,dtype=int)))
        #padding
        padding = np.random.choice(['same','valid'])
        params = [kernel_1,stride_1,padding]
        #print(params,'1d params')
    else:
        #all others
        params = [None]
    print(params,'params output')
    return params

def param_check(layer,params,shape):
    if layer == 4: #if conv then choose which conv based on input length
        first = int(shape[-2])+1
        second = int(shape[-1])+1
        #2d
        #kernel
        if params[0][0] > first:
            params[0][0] = first
        if params[0][1] > second:
            params[0][1] = second
        if params[1][0] > first:
            params[1][0] = first
        if params[1][1] > second:
            params[1][1] = second
        #print(params,'2d params')
    elif layer == 5:
        first = int(shape[-2])+1
        second = int(shape[-1])+1
        #1d
        #kernel
        if params[0] > first:
            params[0] = first
        if params[1] > second:
            params[1] = second
        #print(params,'1d params')
    else:
        #all others
        pass
    print(params,'params output')
    return params

def make_model(model_dictionary,output_layer,layer_dictionary,dimension_dictionary):
    #loop style
    params = []
    layers = []
    #determine the layer and then the shape and params. Then return the params
    X_input = Input(model_dictionary['input_shape'])
    #print(X_input.shape,'input shape')
    layer,param = get_layer(layer_dictionary,dimension_dictionary,X_input.shape,0)
    print(layer,param,'layers,params')
    X = layer_dictionary[0][layer](X_input,param)
    params.append(param)
    layers.append(layer)
    #print(params,layers,'check')
    for i in range(1,model_dictionary['num_total_layers']):
        #Could do a check where if i in N then create the new layer otherwise load layer from a dictionary
        #How to check if the model architecture has been done before? load all layers and then check
        layer,param = get_layer(layer_dictionary,dimension_dictionary,X.shape,i)
        #print(layer,param,'layers,params')
        X = layer_dictionary[i][layer](X,param)
        layers.append(layer)
        #print(params.shape,param.shape,'shapes prior to concat')
        params.append(param)
        #print(params,layers,'check')
    #X = Flatten()(X)
    #print(output_layer,'output')
    X = output_layer(X)
    #X = Dense(1,activation='relu')(X)
    print(layers,params,'layers,params')
    model = Model(inputs = X_input, outputs = X)
    return model,layers,params

def instantiate_model(model_dictionary,output_layer,layer_dictionary,layers,params):
    #loop style
    #determine the layer and then the shape and params. Then return the params
    X_input = Input(model_dictionary['input_shape'])
    #print(X_input.shape,'input shape')
    X = layer_dictionary[0][layer[0]](X_input,params[0])
    #print(params,layers,'check')
    for i in range(1,model_dictionary['num_total_layers']):
        #Could do a check where if i in N then create the new layer otherwise load layer from a dictionary
        #How to check if the model architecture has been done before? load all layers and then check
        X = layer_dictionary[i][layers[i]](X,params[i])
    X = Flatten()(X)
    #print(output_layer,'output')
    X = output_layer(X)
    #X = Dense(1,activation='relu')(X)
    print(layers,params,'layers,params')
    model = Model(inputs = X_input, outputs = X)
    return model,layers,params

def return_new_layer(model_dictionary):
    return  np.argmax(np.random.rand(model_dictionary['num_layer_choices'],model_dictionary['modifications_per_model']),axis=0)
>>>>>>> af08be6... Added genetic algo to taxi_train

def return_layer_config(layer):
    conf = test_layers[random_choice].get_config()
    layer_activation = conf['activation']
    layer_name = conf['name']
    layer_type = layer_name.split('_')[0]
<<<<<<< HEAD

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
=======
    print(layer_name,'layer_name')
    print(layer_type,'layer_type')
    print(layer_activation,'layer_activation')

def get_methods(obj):    
    #test_layers = workers[best_index].model.layers
    #for layer in test_layers:
    #    print(layer.name)
        #print(layer)
    #print(layer[1].get_config())
    method_list_conv = [func for func in dir(obj) if callable(getattr(obj, func))]
    #method_list = [func for func in dir(test_layers[1]) if callable(getattr(test_layers[1], func))]
    print(method_list_conv)
    #print(method_list)
    
def random_initialization(model_dictionary):
    layers = random_layer_initialization(model_dictionary)
    #layers = np.array([[2, 2, 2, 2],[0, 0, 2, 2]]) # for testing purposes
    #layer_params = random_layer_hyperparams(model_dictionary)
    hyperparams = random_hyperparams(model_dictionary)
    return layers,hyperparams

def initialize_models(hyperparams,model_dictionary,layer_dictionary,model_architecture_dictionary,dimension_dictionary,training_dictionary):
    workers = []
    models = []
    layers = []
    layer_params = []
    model_archs = set()
    for i in range(model_dictionary['N']):
        output_layer = copy.deepcopy(layer_dictionary['output_layer'])
        #output_layer = Dense(1,activation='relu')
        model,layer,params = make_model(model_dictionary,output_layer,layer_dictionary,dimension_dictionary)
        key = json.dumps([layer,params,hyperparams[i]])
        print(key,'key')
        #check if key exists, if not add to loss
        while key in model_archs:
            #redo model
            model,layer,params = make_model(model_dictionary,output_layer,layer_dictionary,dimension_dictionary)
            key = json.dumps([layer,params,hyperparams[i]])
        layer_params.append(params)
        layers.append(layer)
        model.compile(optimizer=Adam(lr=model_dictionary['learning_rates'][hyperparams[i]],beta_1=0.9,beta_2=0.999,decay=0),loss='mse')
        if model_dictionary['multigpu'] == True:
            model = multi_gpu_model(model,2)
            model.compile(optimizer=Adam(lr=model_dictionary['learning_rates'][hyperparams[i]],beta_1=0.9,beta_2=0.999,decay=0),loss='logcosh')
        #model.compile(optimizer=np.random.choice(optimizations),loss='mse')
        worker = Worker(model,training_dictionary['num_epochs'],training_dictionary['model_folder'],
                        training_dictionary['val_split'],callbacks_list=training_dictionary['callbacks'])
        model_architecture_dictionary[i] = (layer,params,hyperparams[i])
        model_archs.add(key)
        workers.append(worker)
        models.append(model)
    return workers,models,model_architecture_dictionary,model_archs

def train_models(X,y,workers,models,loss_dictionary,model_architecture_dictionary):
    losses = []
    for worker in workers:
        worker.train(X,y)
        losses.append(worker.loss)
    keys = [json.dumps([model_architecture_dictionary[i][0],model_architecture_dictionary[i][1],model_architecture_dictionary[i][2]]) for i in range(len(models))]
    print(keys,'keys')
    temp_dictionary = dict(zip(keys,losses))
    loss_dictionary = merge_dicts(temp_dictionary,loss_dictionary)
    print(loss_dictionary.items(),'loss_dictionary key values')
    return losses,workers,loss_dictionary

def select_best_model(losses,model_architecture_dictionary):
    best_index = np.argmin(losses)
    return model_architecture_dictionary[best_index]
        
def make_permuted_model(model_architecture_dictionary,model_dictionary,output_layer,layer_dictionary,dimension_dictionary):
    #loop style
    layers,params,_ = model_architecture_dictionary['best']
    layers_to_modify = random.choices(model_dictionary['layers_to_modify'],k=model_dictionary['num_modifications'])
    #Remove layers from possible param selections
    new_params_choices = [model_dictionary['params_to_modify'][i] for i in range(len(model_dictionary['params_to_modify'])) if model_dictionary['params_to_modify'][i] not in layers_to_modify]
    try:
        params_to_modify = random.choices(new_params_choices,k=model_dictionary['num_modifications'])
    except:
        params_to_modify = [-1]
    print(layers_to_modify,params_to_modify,new_params_choices,'layers_to_modify,params_to_modify,new_params_choices')
    print(params,'params')
    print(params[0],params[0][0],'param indexes')
    #determine the layer and then the shape and params. Then return the params
    X_input = Input(model_dictionary['input_shape'])
    #print(X_input.shape,'input shape')
    if 0 in layers_to_modify:
        layers[0],params[0] = get_layer(layer_dictionary,dimension_dictionary,X_input.shape,0)
    elif 0 in params_to_modify:
        params[0] = get_param(layers[0],X_input.shape)
    print(layers[0],params[0],'layers[0],params[0] check')
    X = layer_dictionary[0][layers[0]](X_input,params[0])
    #print(params,layers,'check')
    for i in range(1,model_dictionary['num_total_layers']):
        #Could do a check where if i in N then create the new layer otherwise load layer from a dictionary
        #How to check if the model architecture has been done before? load all layers and then check
        if i in layers_to_modify:
            layers[i],params[i] = get_layer(layer_dictionary,dimension_dictionary,X.shape,i)
        elif i in params_to_modify:
            params[i] = get_param(layers[i],X.shape)
        else:
            #check to make sure the param is correct
            params[i] = param_check(layers[i],params[i],X.shape)
        print(layers[i],params[i],'layers[i],params[0][i] check')
        #print(layer,param,'layers,params')
        X = layer_dictionary[i][layers[i]](X,params[i])
        #print(params,layers,'check')
    #X = Flatten()(X)
    #print(output_layer,'output')
    X = output_layer(X)
    #X = Dense(1,activation='relu')(X)
    print(layers,params,'layers,params')
    model = Model(inputs = X_input, outputs = X)
    return model,layers,params

def permute_model(model_dictionary,model_architecture_dictionary,loss_dictionary,workers,model_archs,dimension_dictionary):
    models = []
    for i in range(model_dictionary['N']):
        output_layer = copy.deepcopy(layer_dictionary['output_layer'])
        model,layer,param = make_permuted_model(model_architecture_dictionary,model_dictionary,output_layer,layer_dictionary,dimension_dictionary)
        if model_dictionary['mod_hyperparams'] == True:
            hyperparam = random.choices(model_dictionary['hyperparams_choices'], k=1)
        else:
            hyperparam = copy.copy(model_architecture_dictionary['best'][2])
        key = json.dumps([layer,param,hyperparam])
        num_trys = 0
        #If model is repeated redo model
        while key in model_archs:
            model,layer,param = make_permuted_model(model_dictionary,output_layer,layer_dictionary,dimension_dictionary)
            if model_dictionary['mod_hyperparams'] == True:
                hyperparam = random.choices(model_dictionary['hyperparams_choices'], k=1)
            else:
                hyperparam = copy.copy(model_architecture_dictionary['best'][2])
            key = json.dumps([layer,param,hyperparam])
            #prevent infinite loop if number of models is too big or num modifications too restricted
            num_trys += 1
            if num_trys == 5:
                print('unable to make unique model')
                break
        #End repetitions
        model_archs.add(key)
        model.compile(optimizer=Adam(lr=model_dictionary['learning_rates'][hyperparam[0]],beta_1=0.9,beta_2=0.999,decay=0),loss='mse')
        if model_dictionary['multigpu'] == True:
            model = multi_gpu_model(model,2)
            model.compile(optimizer=Adam(lr=model_dictionary['learning_rates'][hyperparams[i]],beta_1=0.9,beta_2=0.999,decay=0),loss='logcosh')
        models.append(model)
        workers[i].model = model
        model_architecture_dictionary[i] = (layer,param,hyperparam)
    return models,workers,model_architecture_dictionary,model_archs

def genetic_solution(X,y,model_dictionary,layer_dictionary,training_dictionary,dimension_dictionary,*kwargs):
    #p = Pool(processes=N)
    times_without_update = 0
    model_architecture_dictionary = {}
    loss_dictionary = {}
    if model_dictionary['initialize'] == True:
        ### Initialize models ###
        hyperparams = random.choices(model_dictionary['hyperparams_choices'], k=model_dictionary['N'])
        print(hyperparams,'hyperparams')
        #hyperparams = random_hyperparams(model_dictionary)
        #layers,hyperparams = random_initialization(model_dictionary)
        #print('initial layers \n',layers,'\nhyperparams \n',hyperparams,type(layers),type(hyperparams))
        workers,models,model_architecture_dictionary,model_archs = initialize_models(hyperparams,model_dictionary,
                                           layer_dictionary,model_architecture_dictionary,
                                            dimension_dictionary,training_dictionary)
        losses,workers,loss_dictionary = train_models(X,y,workers,models,loss_dictionary,model_architecture_dictionary)
        best_computed_loss = min(losses)
        print(select_best_model(losses,model_architecture_dictionary),'best model')
        model_architecture_dictionary['best'] = select_best_model(losses,model_architecture_dictionary)
        ### END initialize models ###
    else:
        #UNTESTED
        ### Start with Specific model ###
        #Need to pass the layers,params etc for the specific model so i can pass those to the model_archs set
        layers,params,hyperparams = model_details #passed via kwargs
        output_layer = copy.deepcopy(layer_dictionary['output_layer'])
        #output_layer = Dense(1,activation='relu')
        model = instantiate_model(model_dictionary,output_layer,layer_dictionary,layers,params)
        model.compile(optimizer=Adam(lr=model_dictionary['learning_rate'][hyperparams[i]],beta_1=0.9,beta_2=0.999,decay=0),loss='mse')
        models = [model]
        worker = Worker(model,training_dictionary['num_epochs'],training_dictionary['model_folder'],
                        training_dictionary['val_split'],callbacks_list=training_dictionary['callbacks'])
        workers = [worker]
        key = json.dumps([layer,params,hyperparams[i]])
        model_archs = set(key)
        model_architecture_dictionary[0] = (layer,params,hyperparams)
        losses,workers,loss_dictionary = train_models(X,y,workers,models,loss_dictionary,model_architecture_dictionary)
        model_architecture_dictionary['best'] = model_architecture_dictionary[0]
        ### END ###
    
    ### Genetic Selection ###
    for iteration in range(training_dictionary['iterations']):
        #best_layers,best_params,best_hyperparams = model_architecture_dictionary['best']
        #print(best_layers,best_params,best_hyperparams,'best_layers,best_params,best_hyperparams')
        models,workers,model_architecture_dictionary,model_archs = permute_model(model_dictionary,
                             model_architecture_dictionary,loss_dictionary,workers,model_archs,dimension_dictionary)
        losses,workers,loss_dictionary = train_models(X,y,workers,models,loss_dictionary,model_architecture_dictionary)
        print(losses,'losses')
        best_index = np.argmin(losses)
        print(best_index,'best_index')
        #save best loss for later comparisons
        best_loss = min(losses)
        #save model architecture, might be a better way
        if best_loss < best_computed_loss:
            print('best model update')
            print(model_architecture_dictionary[best_index],'current_best_layers,hyperparams')
            model_architecture_dictionary['best'] = model_architecture_dictionary[best_index]
            best_computed_loss = copy.copy(best_loss)
            times_without_update = 0
        else:
            times_without_update += 1
        if times_without_update == 5:
            print('went 5 rounds without an update... stopping')
            #at some likely optima
            break
    return model_architecture_dictionary['best']
            
def load_json(obj):
    #obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    return np.array(json.loads(obj))
        
#model_dictionary = {0:skip_layer,1:lstm_layer,2:make_dense_relu,3:make_dense_tanh,4:convolutional_block,make_empty_layer}
#I could make the param array an (Number_of_different_filters,Number_of_choices_per_filter,Number_of_models)
#[[filter,stride]]
#Same for hyperparams
#[[Learning_rate,L2,Alpha]]
#[[0.2,0.1,0.01]]
#[[0.02,0.01,0.02]]
#[[0.002,0.001,0.03]]

def genetic_search(X,y,model_dictionary,training_dictionary,layer_dictionary,dimension_dictionary):
    
    
    ### Not used currently ###
    hyperparam_dictionary = {0:0.5,1:0.8}
    param_dictionary = {0:0.5,1:0.8}
    ### End
    return genetic_solution(X,y,model_dictionary,layer_dictionary,training_dictionary,dimension_dictionary)
>>>>>>> af08be6... Added genetic algo to taxi_train

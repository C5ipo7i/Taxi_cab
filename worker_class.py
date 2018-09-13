<<<<<<< HEAD
from keras.models import Model

class Worker(object):
    def __init__(self,model,num_epochs,validation,model_path,verbosity=1,callbacks_list=None,batch_size=None):
=======
class Worker(object):
    def __init__(self,model,num_epochs,model_path,val_split,verbosity=1,callbacks_list=None,batch_size=None):
>>>>>>> af08be6... Added genetic algo to taxi_train
        self.model = model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbosity = verbosity
        self.callbacks_list = callbacks_list
        self.model_path = model_path
<<<<<<< HEAD
        self.validation = validation
    def train(self,X,Y):
        if self.callbacks_list == None:
            self.history = self.model.fit(X,Y,epochs=self.num_epochs,validation_split=self.validation,verbose=self.verbosity)
            self.loss = history['loss'][-1]
            if 'val_loss' in self.history.keys():
                self.val_loss = history['val_loss'][-1]
        else:
            self.history = self.model.fit(X,Y,validation_split=self.validation,epochs=self.num_epochs,verbose=self.verbosity,callbacks=self.callbacks_list)
            self.loss = history['loss'][-1]
            if 'val_loss' in self.history.keys():
                self.val_loss = history['val_loss'][-1]
        
    def evaluate(self,X,Y):
        self.eval_history = self.model.evaluate(X,Y)
        self.eval_loss = eval_history['loss'][-1]
        if 'val_loss' in self.eval_history.keys():
            self.eval_val_loss = eval_history['val_loss'][-1]
=======
        self.val_split = val_split
    def train(self,X,Y):
        if self.callbacks_list == None:
            self.history = self.model.fit(X,Y,epochs=self.num_epochs,verbose=self.verbosity).history
            #print(get_methods(self.history),'methods')
            #print(self.history.keys(),'model loss')
            #print(self.history.on_train_end,'model dict loss')
            #get_methods(self.history.on_train_end)
            
            self.loss = self.history['loss'][-1]
            #self.val_loss = self.history['val_loss']
        else:
            self.history = self.model.fit(X,Y,validation_split=0.15,epochs=self.num_epochs,verbose=self.verbosity,callbacks=self.callbacks_list).history
            self.loss = self.history['loss'][-1]
            self.val_loss = self.history['val_loss'][-1]
        
    def evaluate(self,X,Y):
        val_loss = self.model.evaluate(X,Y)
        return val_loss
>>>>>>> af08be6... Added genetic algo to taxi_train
    def predict_batch(self,X):
        return self.model.predict_on_batch(X)

    def save_model(self):
        self.model.save(self.model_path)
        
    #returns all layers of the loaded model
    def get_layers(self):
        layers = []
        for layer in self.model.layers:
            layers.append(layer)
        return layers
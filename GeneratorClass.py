import numpy as np

class DataGenerator(keras.utils.Sequence):
    def __init__(self,list_IDs,labels,batch_size,dim,n_channels=1,n_classes=10, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.labels = labels
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        #Generate one batch of data
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        #Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        #generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X,y

    def on_epoch_end(self):
        #Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self,list_IDs_temp):
        #generates data containing batch_size samples. X : (n_samples, *dim, n_channels)
        #Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        #generate data
        for i, ID in enumerate(list_IDs_temp):
            #Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            #Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
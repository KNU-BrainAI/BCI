from itertools import count
import sys
import numpy as np
import json
from scipy.signal import butter,lfilter
# EEGNet-specific imports
import tensorflow as tf
import mne
from tensorflow.keras import utils as np_utils
from sklearn.model_selection import train_test_split
sys.path.append('C:/Users/PC/Desktop/BCI')
from Model import Multi_DS_EEGNet
import glob
import matplotlib as plt
#show device placement
#tf.debugging.set_log_device_placement(True)
import socket

#saving result path
msg_dir =  "D:/Artigence/BCI/Server/msg_buffer.txt"
# while the default tensorflow ordering is 'channels_last' we set it here
class Network_class:
    def __init__(self):
        self.signal = None
        self.input_signal = None
        self.count = 0
        self.X_train = []
        self.Y_train = []
        self.Trained_model = None
        self.cl_flag = 0
        self.avg_flag = 0
        self.avg_count = 0
        self.sum = 0
        self.sum0 = 0
        self.sum2 = 0
        self.average = 0
        self.average2 = 0
    def get_count(self):
        self.count = self.count +1
        
        return self.count * 3

    def Network(self,in_sig):
        """
        input에 preprocessing-> mne
        overlapping-> input 1~5
        """
        time = self.get_count()
        print(time)
        
        if time == 3:
            print("-----Think Left Hand 1.5 min-----")
        elif 15<time<=105:
            self.signal = in_sig 
            self.Train_data(self.signal,0)
        elif time == 108:
            print("30sec Break Time")

        elif time==129:
            print("-----Think Right Hand 1.5 min-----")
        elif 138<time<=228:
            self.signal = in_sig 
            self.Train_data(self.signal,1)
           
        elif time==231:
            print("-----Think Foot 1.5 min-----")
        elif 231<time<=291:
            self.signal = in_sig
            self.Train_data(self.signal,2)
           
        elif 291<time<=294:
            print("-----Training Data-----")
            print("X_train_length : ", len(self.X_train))
            print("Y_train_length : ", len(self.Y_train))
            self.Trained_model = self.Training(self.X_train,self.Y_train)
            print("-----Training End-----")
            self.cl_flag = 1
        """
        if(time==18):
            self.cl_flag = 1
            self.Trained_model = self.Load()
        """
        if(self.cl_flag == 1):
            if(self.avg_flag == 0):
                if(self.avg_count == 7):
                    self.avg_flag = 1
                self.signal = in_sig 
                # list -> np array (input, channel, sample, kernel)
                self.signal = np.reshape(self.signal,(1,32,768,1))
                self.input_signal = self.signal
                self.averaging(self.Trained_model,self.input_signal)
                print("temp_sum:", self.sum, "avg_count", self.avg_count)
                self.average = self.sum/self.avg_count
                self.average2 = self.sum2/self.avg_count
            else:
                self.signal = in_sig 
                # list -> np array (input, channel, sample, kernel)
                self.signal = np.reshape(self.signal,(1,32,768,1))
                self.input_signal = self.signal
                #print(self.input_signal.shape)
                self.classification(self.Trained_model,self.input_signal)
          
    
    def Train_data(self,data,label):
        data = np.array(data) 
        self.X_train.append(data)
        self.Y_train.append(label)
    def Load(self):
        model = Multi_DS_EEGNet(nb_classes=3, Chans=32, Samples=768,
                    dropoutRate=0.5, kernLength=128, F1=8, D=2, F2=16)
        model.load_weights("C:/Users/PC/Desktop/data/checkpoints/epoch_002.ckpt")
        return model
    def Training(self, X_train_in, Y_train_in):
        kernels, chans, samples = 1, 32, 768
        X_train = np.array(X_train_in)
        Y_train = np.array(Y_train_in)
        print(X_train.shape,Y_train.shape)
        ## convert data to NHWC (trials, channels, samples, kernels) format. Data
        X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        Y_train = np_utils.to_categorical(Y_train)
        # take 50/25/25 percent of the data to train/validate/test
        print(X_train.shape,Y_train.shape)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.8, shuffle=True,
                                                        random_state=1004)
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.8, shuffle=True,
                                                        random_state=1004)
        
        model = Multi_DS_EEGNet(nb_classes=3, Chans=32, Samples=768,
                    dropoutRate=0.5, kernLength=128, F1=8, D=2, F2=16)
        
        model.compile(loss='categorical_crossentropy', optimizer='adam',
            metrics=['accuracy'])
        checkpoint_path = "C:/Users/PC/Desktop/data/checkpoints/epoch_{epoch:03d}.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=True, period=1,
                                                verbose=1, save_best_only = True)
        class_weights = {0: 1, 1:1, 2:1}
        model.fit(X_train, Y_train, batch_size=2, epochs=10,
                verbose=2, validation_data=(X_val, Y_val), shuffle=True,
                class_weight=class_weights, callbacks=[cp_callback])
        
        probs = model.predict(X_test)
        preds = probs.argmax(axis=-1)
        print(probs)
        print(preds)
        print(Y_test.argmax(axis=-1))
        acc = np.mean(preds == (Y_test.argmax(axis=-1)))
        print("Classification accuracy: %f " % (acc))
        return model
   
    def classification(self,model,input):
        input = input 
        probs = model.predict(input)
        print(probs) #
        
        if(probs[0][2] > 0.7 ):
            result = 2     
        else: 
            if abs(probs[0][0] - probs[0][1]) < 0.3:

                if((probs[0][0] -(self.sum0/self.avg_count)) > (probs[0][1] -(self.sum/self.avg_count))):
                    result = 0
                    print(probs[0][0] -(self.sum0/self.avg_count))
                    print(probs[0][1] -(self.sum/self.avg_count))
                else: 
                    result = 1
                    print(probs[0][0] -(self.sum0/self.avg_count))
                    print(probs[0][1] -(self.sum/self.avg_count))
            else: 
                if(  (self.sum0/self.avg_count) > (self.sum/self.avg_count) ):
                    if ( (probs[0][0]/(self.sum0/self.avg_count)) >1.2 ):
                        result = 0
                    else: 
                        result = 1
                else:
                   if ( (probs[0][1]/(self.sum/self.avg_count)) > 1.2 ):
                       result = 1
                   else: 
                        result = 0
        # find max value index
        preds = probs.argmax(axis = -1)
        print(preds)

        print("Classification result: Left = 0, Right = 1, Foot = 2 =>" , result)


        f = open(msg_dir, "a")
        #f.write("="*50 + "\n") 
        #f.write(str(signal) + "\n")
        f.write(str(result) + "\n")
        f.close()

    def averaging(self,model,input):
        input = input 
        probs = model.predict(input)
        print(probs) #
        
        self.sum0 = self.sum0 + probs[0][0] 
        self.sum = self.sum + probs[0][1]
        self.sum2 = self.sum2 + probs[0][2]
        self.avg_count = self.avg_count + 1 
        print("Averaging")


        # find max value index
        
        


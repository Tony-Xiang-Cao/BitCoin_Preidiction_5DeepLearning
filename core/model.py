''' 
Define models to train LSTM, RNN and WaveNet
'''
import os
import time
import logging
import datetime as dt
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, Dense, Dropout, LSTM, SimpleRNN, TimeDistributed
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


class LSTMTimeSeriesModel:
    '''
    Class for building the LSTM
    '''
    def __init__(self):
        self.model = Sequential()
    
    def load_model(self, filepath):
        '''
        Loading the model from a filepath
        '''      
        logging.info(f"Loading model from {filepath}")
        self.model = load_model(filepath)
        
    #3 LSTM layer
    def build_model_original(self, config):

        logging.info("[MODEL]: Building model...")
        now = time.time()

        #Reading model set-up configuration file
        for layer in config['model']['layers']:
            units = layer['units'] if 'units' in layer else None
            dropout = layer['dropout'] if 'dropout' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            seq_len = layer['seq_len'] - 1 if 'seq_len' in layer else None         
            num_features = layer['num_features'] if 'num_features' in layer else None
            layer_type = layer['type'] if 'type' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            
            if layer_type == 'Dense':
                self.model.add(Dense(units=units, activation=activation))
            elif layer_type == "LSTM":
                self.model.add(LSTM(units=units, 
                                    activation=activation, 
                                    input_shape=(seq_len, num_features), 
                                    return_sequences=return_seq
                                   ))
            elif layer_type == "Dropout":
                self.model.add(Dropout(rate=dropout))
                
        self.model.compile(loss=config['model']['loss'], optimizer=config['model']['optimizer'])
        
        time_taken = time.time() - now    
        logging.info(f"Model Building complete in {time_taken//60} min and {(time_taken % 60):.1f} s")
    

    #WaveNet, stacking 12 layers of CONV1D
    def build_model_wavenet(self, config):
        seq_len = config['data']['sequence_length'] - 1 
        num_features = config['data']['num_features']
        logging.info("[MODEL]: Building model...")
        self.model.add(layers.InputLayer(input_shape=[seq_len,num_features]))
        for rate in (1, 2, 4, 8) * 3:
            self.model.add(layers.Conv1D(filters=20, kernel_size=2, padding="causal",
                                  activation="relu", dilation_rate=rate))
        self.model.add(layers.Conv1D(filters=10, kernel_size=1))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1,activation='linear'))  
        #Use Adadelta optimizer to help converging during training and 
        #addresses issue of continuously reduced learning rates.
        self.model.compile(loss='mse',optimizer='Adam', metrics =['mae'] )
 
    
    # 2 RNN layers
    def build_model_RNN(self,config):
        seq_len = config['data']['sequence_length'] - 1 
        num_features = config['data']['num_features']
        logging.info("[MODEL]: Building model...")
        self.model.add(SimpleRNN(100, input_shape=(seq_len, num_features), return_sequences = True))
        self.model.add(SimpleRNN(100, return_sequences = False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='linear'))
        opt = Adam(learning_rate=1e-4, clipnorm = 1) 
        self.model.compile(loss='mse',optimizer=opt)
   
        
    def train(self, x_train, y_train, config):
        '''
        Function to train model
        '''
        epochs = config["training"]["epochs"]
        batch_size = config["training"]["batch_size"]
        save_dir = config["model"]["save_dir"]
        
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, **config["model"]["checkpoint_params"]),            
            ReduceLROnPlateau(**config["model"]["reduce_lr_params"]),          
            EarlyStopping(**config["model"]["early_stopping_params"]),  
        ]
        logging.info("[MODEL]: Training started")
        history = self.model.fit(
                    x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=config["training"]["val_split"],
                    callbacks=callbacks        
                )
        self.model.save(save_fname)
        
        logging.info(f"Model training completed. Model saved to {save_fname}")
        
        return history
    
    def predict_point_by_point(self, data):
        '''
        Making one prediction for each sequence
        '''
        logging.info('[MODEL]: Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        
        return predicted
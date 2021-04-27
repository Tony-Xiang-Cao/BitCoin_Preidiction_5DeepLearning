''' 
Sequence-to-Sequence models using LSTM and GRU layers
End to end load, train, test, predict, calculate return
'''

import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
from keras.regularizers import L1L2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras import layers
import os
from sklearn import metrics

from core.detrendPrice import detrendPricing
from core.WhiteRealityCheckFor1 import bootstrap

def sequence_to_sequence(s2smodel,datafile,train_split,test_split,load_model, modelfile):

  def to_supervised(data, n_input, n_out):
    n_input=n_input+n_out
    X = list()
    in_start = 0
	  # step over the entire history one time step at a time
    for _ in range(len(data)):
		  # define the end of the input sequence
      in_end = in_start + n_input
      out_end = in_end + n_out
		   #ensure we have enough data for this instance
      if out_end < len(data):
         X.append(data[in_start:in_end, :]) #process all columns
		  # move along one time step
      in_start += 1
    return np.array(X)


  # convert history into a normalized cube for series to series prediction
  def to_supervised_normalized(data, n_input, n_out):
	  f = 10
	  small = .001 #to prevent division by zero error
	  n_input=n_input+n_out
	  X = list()
	  in_start = 0
	  # step over the entire history one time step at a time
	  for _ in range(len(data)):
		  # define the end of the input sequence
		  in_end = in_start + n_input
		  out_end = in_end + n_out
		   #ensure we have enough data for this instance
		  if out_end < len(data):
			  X_input = data[in_start:in_end, :] #processing all columns
			  df = pd.DataFrame(X_input)
			  df = ((df/(df.iloc[0,:]+small))-1)*f
			  X_input = df.values
			  X.append(X_input)
		  in_start += 1
	  return np.array(X)
  
  def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

  data_dir= os.path.join('data/',datafile)  
  dataset = pd.read_csv(data_dir, header=0, infer_datetime_format=True, parse_dates=['Date'], index_col=['Date']) 
  dataset['hash_rate'] = dataset['hash_rate'].astype(float)
  dataset['difficulty'] = dataset['difficulty'].astype(float)
  data = dataset.values
  target_data = data[:,0:1] #target data needs to be an array, and leftmost column in data
  #target_data = np.expand_dims(dataset.Open.values, axis=1) if you want to select the target by name
  #target_data.shape

  n_steps = 20 #LSTM input lookback
  fut_steps = 4 #future steps to predict

  split1 = int(np.round(len(data)*train_split)) #train end
  split2 = int(np.round(len(data)*(1-test_split))) #validation end (make sure you do not have a crash in the middle of the train + validation data)   

  series = to_supervised_normalized(data, n_input=n_steps, n_out=fut_steps) #with normalization
  #series = to_supervised(data, n_input=n_steps, n_out=fut_steps)

  X_train = series[:split1, :n_steps]
  X_valid = series[split1:split2, :n_steps]
  X_test = series[split2:, :n_steps]
  n_features = X_train.shape[2]

  Y = np.empty((len(series), n_steps, fut_steps))
  yseries = to_supervised_normalized(target_data, n_input=n_steps, n_out=fut_steps) #with normalization
  #yseries = to_supervised(target_data, n_input=n_steps, n_out=fut_steps)

  for step_ahead in range(1, fut_steps + 1):
    Y[..., step_ahead - 1] = yseries[..., step_ahead:step_ahead + n_steps, 0]
  Y_train = Y[:split1] 
  Y_valid = Y[split1:split2] 
  Y_test = Y[split2:] 

  verbose, epochs, batch_size = 1, 40, 16 #1,150,16
  #Build S2S LSTM model
  if s2smodel == 'LSTM':
    model = keras.models.Sequential([
        keras.layers.LSTM(200, return_sequences=True, input_shape=[n_steps, n_features],
                          dropout=0.05, recurrent_dropout=0.1, kernel_regularizer=L1L2(0.0, 0.0)),
        keras.layers.LSTM(200, return_sequences=True), 
        keras.layers.Dropout(0.05),
        keras.layers.LSTM(200, return_sequences=True),
        keras.layers.Dropout(0.05),
        keras.layers.TimeDistributed(keras.layers.Dense(fut_steps))
    ])
  #Build S2S GRU model
  elif s2smodel == 'GRU':
     model = keras.models.Sequential([
        keras.layers.Conv1D(filters=20, kernel_size=n_steps, strides=2,
                             padding ='valid', input_shape=[n_steps,n_features]),
        keras.layers.GRU(20, return_sequences=True), 
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.Dropout(0.05),
        keras.layers.TimeDistributed(keras.layers.Dense(fut_steps))
    ])
  else:
    print('Warning:wrong model name!')


  if load_model == True:
    #load pre_trained model
    model_dir= os.path.join('saved_models/',modelfile)  
    model = keras.models.load_model(model_dir, custom_objects={'last_time_step_mse': last_time_step_mse})
    
  else:
    # train a seq2seq model if no model is loaded
    callbacks_list = [
    ModelCheckpoint(
        filepath="LSTM-weights-best.hdf5",
        monitor = 'val_loss',
        save_best_only=True,
        mode='auto',
        verbose=1
       ),
    ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.1,
        patience=2,
        verbose=1
        ),
    EarlyStopping(
        monitor = 'val_loss',
        patience=5,
        verbose=1
        )
    ]

    model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                        callbacks=callbacks_list,shuffle=True,
                        validation_data=(X_valid, Y_valid))

    train_loss_values = history.history["loss"] #training loss
    val_loss_values = history.history["val_loss"] #validation loss
    epochs = range(1,len(train_loss_values)+1)

    plt.clf()
    df = pd.DataFrame(train_loss_values, columns=["training_loss"], index=epochs)
    df["validation_loss"] = val_loss_values
    title="training loss vs validation loss"
    df[['training_loss','validation_loss']].plot(grid=True,figsize=(8,5))
    plt.show()


    print("val_loss_value: ", val_loss_values[-1])

  # end of training
    
  # make a prediction on trained or loaded model

  predictions = list()

  

  for i in range(X_test.shape[0]):
    # predict the week
    input_x = X_test[i,:,:]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape(1, input_x.shape[0], input_x.shape[1])   
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0:1, -1:, :][0][0]
    yhat_sequence = yhat
    # store the predictions
    predictions.append(yhat_sequence)
  # evaluate predictions days for each week
  predictions = np.array(predictions)

  #plot prediction vs true target data
  plt.plot(predictions[:,0], label='Prediction')
  plt.plot(Y_test[:, -1, 0], label='True Data')
  plt.legend()
  plt.show()
  #root mean squared error of the prediction
  error = metrics.mean_squared_error(Y_test[:, -1, 0], predictions[:,0], squared=False)
  print("rmse:", error)

  series = to_supervised(data, n_input=n_steps, n_out=fut_steps) #with normalization
  #series = to_supervised(data, n_input=n_steps, n_out=fut_steps)
  X_train_unnor = series[:split1, :n_steps]
  X_valid_unnor = series[split1:split2, :n_steps]
  X_test_unnor = series[split2:, :n_steps]

  test_x_prices = X_test_unnor[:,:,:1] #get the Open prices (leftmost column, =the target =the Open )
  todays_price = test_x_prices[:,-1:,:].reshape(-1)  #getting the latest Open price, reshaping
  df = pd.DataFrame(todays_price, columns=['todays_price'])
  df['predicted_firstday_price'] = predictions[:,0] #LSTM predict
  df['mkt_returns']=(df.todays_price-df.todays_price.shift(1))/df.todays_price.shift(1)

  df['signal'] = np.where(df.predicted_firstday_price>=0,1,0) 
  #df['signal'] = np.where(df.predicted_firstday_price<0,-1,df.signal) #for shorting

  df['system_returns'] = df.signal.shift(1)*df.mkt_returns
  df['system_equity']=np.cumprod(1+df.system_returns) -1
  df['mkt_equity']=np.cumprod(1+df.mkt_returns) -1

  title="system_equity"
  df[['system_equity','mkt_equity']].plot(grid=True,figsize=(8,5))
  plt.ylabel("system_equity")
  plt.show()


  system_cagr=(1+df.system_equity.tail(n=1))**(252/df.shape[0])-1
  df.system_returns= np.log(df.system_returns+1)
  system_sharpe=np.sqrt(252)*np.mean(df.system_returns)/np.std(df.system_returns)
  print("system_cagr: ",system_cagr*100)
  print("system_sharpe: ",system_sharpe)

  mkt_cagr=(1+df.mkt_equity.tail(n=1))**(252/df.shape[0])-1
  df.mkt_returns= np.log(df.mkt_returns+1)
  mkt_sharpe=np.sqrt(252)*np.mean(df.mkt_returns)/np.std(df.mkt_returns)
  print("mkt_cagr: ",mkt_cagr*100)
  print("mkt_sharpe: ",mkt_sharpe)

  #white reality check
  #Detrending Prices and Returns and white reality check
  #Detrend prices before calculating detrended returns
  df['Det_todays_price'] = detrendPricing(df['todays_price']).values 
  #these are the detrended returns to be fed to White's Reality Check
  df['DetRet'] = (df['Det_todays_price']-df['Det_todays_price'].shift(1))/abs(df['Det_todays_price'].shift(1)) 
  df['DetStrategy'] = df['DetRet'] * df['signal'].shift(1)
  bootstrap(df['DetStrategy'])

  return system_cagr, system_sharpe 

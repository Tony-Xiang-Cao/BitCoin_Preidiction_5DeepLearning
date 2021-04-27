# Bitcoin Prediction with 5 Deep Learning Models
## APS 1052 Final Project 2021 Spring
 **Author:**

Xiang Cao

**Date:** 23-Apr-2021

## Project Executive Summary: 
In this project, We compared 5 different deep learning models for prediction of Bitcoin Price. Those 5 models are mostly Recurrent Neural Network based that specialize in predicting sequential time series data. The original Bitcoin dataset is collected over the period from 2013-10-01 to 2020-10-09 with a total of 2556 samples and 16 features that tracks Bitcoin and financial market changes over time. We added a look back window of 20 days to the Bitcoin dataset and constructed a 3-dimensional time series data with temporal dimension to predict Bitcoin price on the 21st day. We evaluate 5 deep learning models on the dataset: **LSTM, RNN, WaveNet, Sequence-to-sequence LSTM and GRU+CNN**. Among those 5 models, the LSTM, RNN, and WaveNet are sequenc-to-vector models that output 1 prediction, while the other 2 models are sequqnce-to-sequence models that output a sequence of 4 predictions at each time step. 

We also did feature engineering by adding additional 8 different features including 5 Commodity Futures (COT) features and 3 related stock price features(Ticker: QQQ, FINX, RIOT). We compare the performance before and after adding the additioanl 8 features and found most deep learning model perform better with more related features. CAGR and Sharpe ratio is calculated with an only-long trading strategy based on the model prediction. After calculating White's reality check, most of the result have a p-value less than 0.1 threshold and we thus reject the null hypothesis and accept the model prediction.

All 5 deep learning models beat market performance by a significant margin.  Over the evaluation of all 3 dataset, WaveNet has constantly outperform other recurrent neural network based models, which is consitent with Geron's finding in his book Hands-on Machine Learning with Scikit-Learn, Keras & Tensorflow. The outstanding performance may due to its capability of storing long sequence, which enables learning historical event years ago. In conclusion, we recommend using WaveNet model, and add additional featuers to dataset for Bitcoin price prediction.  

## Code Structure
1. **BitCoin_Preidiction_5DeepLearning.ipynb**

    This note book is the main program which will load all packages and supporting functions, run training and testing of all 5 models.

    - The notebooks started with several functions definition including loading dataset and configuration, training model, loading pretrained models, and make prediction and evaluate results. Those functions are for LSTM, RNN and WaveNet models. The two seq-to-seq models have thier entire workflow written within their function.
    - Load the original bitcoin dataset,train, predict and evaluate all 5 deep learning models.
    - Load the 2017-2020 bitcoin subset, train, predict and evaluate all 5 deep learning models.
    - Load the 2017-2020 augmented bitcoin dataset, train, predict and evaluate all 5 deep learning models.
2. **data  folder**

    This folder contains all related dataset. 
    - btc_dataset.csv is the original 2013-2020 bitcoin dataset, created by previous author. 
    - **add_COT_QQQ.ipynb** is our code for the feature engineering and data cleaning of COT data and realted stock data.
    - 2017,2018,2019,2020.csv are COT dataset downloaded from the Commodity Futures Trading Commission (CFTC) website. 
    - QQQ_FINX_RIOT.csv is stock price features downloaded from WRDS databse. 
    - btc_dataset_2017-2020.csv is a subset of btc_dataset.csv, selecting only the range from Dec 2017 to Oct 2020 when COT data is available. 
    - btc_dataset_COTQQQ.csv is the augemented dataset from Dec 2017 to Oct 2020 with 8 additional features.

3. **core folder**

    This folder contains 5 machine learning models, and supporting code for the program.
    - **model.py** defines LSTM,RNN, WaveNet model architectures.
    - **Sequence2Sequence.py** defines end-to-end workflow of Sequence-to-Sequence LSTM and GRU model(load -> preprocess -> train -> test -> plot).
    - **dataloader.py** load dataset, creating temporal sequence,and generate a 3D time series dataset.
    - setup.py & utils.py supporting functions
    - WhiteRealityCheckFor1.py & detrendPrice.py Functions for White's reality check

4. **saved model foler**

    This folder stores model previously trained model in tensorflow format.

5. **config_xx.json**

    These 3 json files stores configuration for each dataset, such as number of features and train-test split ratio.

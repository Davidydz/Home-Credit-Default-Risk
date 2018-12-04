# Home-Credit-Default-Risk
## Overview of the model
The first step is to clean the data as well as extracting the features. There are two ways to extract data, it result in two dataset. Then I apply the LightGBM on each of the dataset to make prediction, and then blend these two together to achieve a better result.


## 1. Data preparation
a. There are two ways to extract features. The first one is given by Ann Antonova (https://www.kaggle.com/aantonova). Each client has a credit history which can be viewed as a combination of several time series. For each a single dimension time series, several features will be extracted(). This result in a large number of features including the redundant ones. But only  those helpul in prediction will be keeped. 

b. The way that I deal with this is a little bit different. Instead of treating the credit history as multi-dimensional time series, I treated it as samples coming from the same distribution. So for each client, I assume their credit history follow the same distribution with different mean value and variance covariance matrix. And for each client, I will get the mean value and the correlation as a summary of his previous history. Then I filter out those features that does not help prediction, and apply LightGBM to make prediction.


## 2. Application of Machine Learning Algorithm
The algorithm that I choose is LightGBM, and hyper-parameters are selected using Bayesian-Optimization. 

## 3. Blend Model 
After obtaining the prediction from my model and Ann's model, I blended these two models together using simple logistic regression. The choice of using logistic regression is to avoid overfitting.

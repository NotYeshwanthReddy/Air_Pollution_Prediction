# Air_Pollution_Prediction

## 1. INTRODUCTION:
With the development of industry, air pollution has become a serious problem. It is very important
to create an air quality prediction model with high accuracy and good performance. Globally, the
problem of poor air quality as an environmental risk has grown significantly. As a result, predicting
and assessing ambient air quality has grown in importance. In general, the term "air quality" refers
to the measurement of clean air in a certain area. It is calculated by measuring several atmospheric
pollution indicators. The concentration of air pollutants is estimated using traditional methods,
which are computationally intensive. Additionally, these approaches are unable to make sense of
the wealth of information at hand. The suggested work offers a deep learning method for
quantifying and forecasting ambient air quality in order to address this problem. Long Short-Term
Memory (LSTM) is a unique type of structured memory cell that is used in a framework based on
recurrent neural networks (RNNs) to do air quality prediction. In this work, we proposed LSTM and
GRU (Gated Recurrent Unit) to capture the dependencies in various pollutants and to perform air
quality predictions.


## 2. SURVEY:
Numerous statistical methods are used to predict the air quality. In this study, we look into the
abilities of different deep learning models to predict the concentration of PM2.5. As a result, we
choose to employ the LSTM, GRU and DeepAR. Then, we succinctly outline each network:

### 2.1. LSTM [Long Short-Term Memory]
LSTM is an improved approach to conventional RNN. By including a memory block, LSTM resolves
the RNN's vanishing gradient issue. With a constant error carousel (CEC) value of 1, a memory block
is a complicated processing unit that has at least one memory cell as well as a few multiplicative
gates serving as its input and output. The memory block does not receive any outside signal values,
which causes the error value to become active for the duration of the time step. The entire
operation of the memory block is under the control of the multiplicative gates. An input gate
regulates the flow of input into a memory cell by controlling the activation of the cell. Three gates
make up an LSTM: an input gate that decides whether to accept fresh data, a forget gate that
eliminates unimportant information, and an output gate that selects the information to be
produced. These three gates operate in the 0 to 1 range and are analogical gates based on the
sigmoid function.
Fig. Below shows these three sigmoid gates. The cell state is represented by a horizontal line that
passes through it.

![image](https://github.com/NotYeshwanthReddy/Air_Pollution_Prediction/assets/31570514/27eb7153-9f2d-48ba-b1cd-a3cfcae2f659)

The LSTM architecture estimates an output ot=(ot 1,ot 2,ot 3,...,otT-1,ot T) by updating three
multiplicative units (input I output (op), and forget gate (fr)) on the memory cell with continuous
write, read, and reset operations on the memory cell (mc) from time t=1 to T.

![image](https://github.com/NotYeshwanthReddy/Air_Pollution_Prediction/assets/31570514/dd75c106-a529-4b72-8dbf-ea5804cd625d)

Since LSTMs are frequently used for sequential analysis, they can be trained to forecast air quality index levels for the upcoming hour or even the upcoming month using the historical data gathered by sensors at different weather stations.

### 2.2. GRU [Gated Recurrent Unit]:
An addition to the LSTM network is GRU. Update and reset gates make up the system. They all include balancing the data flow within the unit. The GRU receives time series data from AirNet as input. The GRU, or gated recurrent unit, is an improvement over the traditional RNN [33] and is incorporated into RNN. It is comparable to an LSTM unit. The reset and update gates make up the GRU unit. The GRU architecture is shown in Figure below. While the update gate is used to choose the number of the candidate activation that updates the cell state, the reset gate is intended to erase the previous state between the previous activation and the next candidate activation.

![image](https://github.com/NotYeshwanthReddy/Air_Pollution_Prediction/assets/31570514/d8f6b382-84ae-4ffc-bfa8-5c530a3901ab)

The update gate is used to regulate how much data from hidden states in the past is carried over into the current state. More information about past states is brought in as the update gate value increases. The amount to which the information from earlier stages is discarded is controlled by the reset gate, and the smaller the value of the reset gate, the more it is ignored. As a result, long-term dependencies are accompanied by the activation of update gates, but short-term relationships are typically captured with frequent activation of reset gates.

![image](https://github.com/NotYeshwanthReddy/Air_Pollution_Prediction/assets/31570514/87f888e4-7daf-4b69-84c0-93c8a0e58ef6)

### 2.3. DeepAR: 
DeepAR is a forecasting algorithm used for forecasting scalar one dimensional time-series data. It’s a recursive neural network designed by Amazon research group. Classic forecasting methods like ARIMA and ETS can only be trained on individual time-series data and forecast it. However, DeepAR is designed to learn multiple related time series and forecasts using the combined knowledge of all the related data. When a dataset contains multiple related time series, DeepAR outperforms the traditional ARIMA and ETS algorithms.
The model has multiple tuneable hyperparameters like context_length, prediction_length, learning_rate, dropout_rate, embedding_dimension, num_cells and num_layers. The context_length decides the number of past records which the model has visibility on. And prediction_length decides the number of future records the model can predict. Amazon suggests to keep prediction_length less than or equal to context_length to ensure model’s predictions are close to real values.


### 2.4. Auto-Regression:
![image](https://github.com/NotYeshwanthReddy/Air_Pollution_Prediction/assets/31570514/3ee44626-952d-4ca6-bfa4-a9771a6541ed)
Auto-regression is the method in which a linear combination of past values of the variable are used
for prediction/forecasting. An autoregressive model of an order ‘p’ can be written as:
𝑦 𝑡 = 𝑐 + ϕ1 𝑦 𝑡−1 + ϕ2 𝑦 𝑡−2 . . . . . + ϕ𝑝 𝑦 𝑡−𝑝 + ε𝑡
Here 𝑦 is the prediction value at time stem t. is the white noise and is the parameter which is
𝑡 ε 𝑡 ϕ𝑝
multiplied with 𝑦 . This is like a multiple regression but with a lagged values of . We can refer to
𝑡𝑦𝑡
this model as AR(p) which represents an Auto-regressive model of order ‘p’.
Few things we need to consider while building an auto-regressive model are the length of the
context window and the length of the prediction window. An auto-regressive model uses its
previous prediction as input for its future prediction. i.e. prediction 𝑦 is used as input for
𝑡−1 predicting 𝑦 .𝑡
The context window defines the length of the linear combination of (𝑦 , . . . , ) previous
𝑡−1 𝑦 𝑡−2 𝑦 𝑡−𝑐
predictions to be used for new prediction 𝑦 . And the prediction window defines the number of
𝑡 time steps the auto-regressive model can be called to predict the future. An auto-regressor with
prediction window 10 can predict up-to 10 days in the future considering each time step accounting for 1 day

## 3. IMPLEMENTATION:
### 3.1. Visualizing of Time Series
When evaluating a time series model's stability over time, rolling analyses are frequently used. An
important presumption when using a statistical model to analyse financial time series data is that
the model's parameters won't change over time. To illustrate the outcome of the anticipated air
quality plot, we use matplotlib. The following graphs show the Air quality of India (2017-2022) and
Air quality of 1 year (2017-2018). We have chosen to average the data from hourly basis to daily
basis. This illustration gives us an idea of how the patterns in data change over time.
![image](https://github.com/NotYeshwanthReddy/Air_Pollution_Prediction/assets/31570514/9001dd42-25b6-4191-9e88-13738897fc59)

### 3.2. Prediction using LSTM Model:
It is a sequential model, which is made up of a linear stack of layers. Then we introduced dense.
which is the standard deeply coupled neural network layer. It is used to change the output vector's
dimensions during backpropagation. Next, we establish our RNN using a regression model. Read the
sequential data in and add it to the regressor model to achieve this.
The neural network receives the input and is trained for prediction using random biases and
weights. A sequential input layer, three LSTM layers, and a dense output layer with a linear
activation function make up the LSTM model. The output value produced by the RNN's output layer
is contrasted with the desired value. The backpropagation algorithm reduces error, or the
discrepancy between the desired output value and what is actually produced.

Output Value:
The testing scores obtained for LSTM model are as follows:
```
Test Score: 4.82 RMSE
Test Score: 0.93414 R2
```
The graph obtained below is plotted against models predictions over training data, testing data and
actual data.
![image](https://github.com/NotYeshwanthReddy/Air_Pollution_Prediction/assets/31570514/56e0c712-0c5a-4569-95f6-749774389c71)
We also employ auto-regression technique over the existing LSTM model with length of context window as
100 and length of prediction window to be 365. This model takes 100 days data as input and can predict
up-to 365 days in the future which is a full 1 year prediction based on past 3 months data. The following
graph shows the output of the auto-regression model trying to predict 365 days into the future using past
data.
As the context window is 100, the model’s predictions are reliable up-to 100 days into the future. i.e. a
Prediction window with 100 length is reliable. However further predictions are made based on the previous
predictions which will result in higher deviation from the original data.

![image](https://github.com/NotYeshwanthReddy/Air_Pollution_Prediction/assets/31570514/84ecc2a9-1982-4fc2-b29e-89d49504c6ab)

## 3.3. Prediction using Bi-directional LSTM:
Similarly, we carried out the experiment using bidirectional LSTM model with 1 layer as well as an
add on and the outputs turned out to be as follows:
The testing scores and root mean squared errors obtained for this model are:
Test Score: 4.80 RMSE
Test Score: 0.93461 R2
The graph obtained below is plotted against predictions over training and testing data overlapped
over original values.

![image](https://github.com/NotYeshwanthReddy/Air_Pollution_Prediction/assets/31570514/b35590af-f584-47a2-b7af-20dc3f818845)

Similarly, we also employed an Auto-regressive approach to the existing Bi-LSTM model and the following
graph represents the prediction of 365 days based on the past data.

![image](https://github.com/NotYeshwanthReddy/Air_Pollution_Prediction/assets/31570514/a7bcba1f-7029-41d9-a80a-df0d0e2b7982)

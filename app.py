import numpy as np
import pandas as pd
import streamlit as stm
from datetime import datetime
from keras.models import load_model
import yfinance as yf 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler


stm.set_page_config(layout = "wide")
stm.title("Stock Analysis App")

stocks = ["SBIN.NS","RELIANCE.NS","TCS.NS","EICHERMOT.NS","MRF.NS"]
select_stock = stm.selectbox("Select stock for prediction", stocks)

no_years = stm.slider("Select number of years:", 1, 10)

# inval = 160 if 5 < no_years else 60

end = datetime.now().date()
start = datetime(end.year - no_years, end.month, end.day).strftime("%Y-%m-%d")

df = yf.download(select_stock, start, end)
df = df.reset_index()


#Information about Stocks

stm.subheader(f"Stock data from {start} to {end}")
stm.write(df.describe())

#Data visualization

#Moving Avg.

stm.subheader(f"{select_stock}'s moving average chart: Closing price vs Time")

MA10 = df.Close.rolling(10).mean()
MA20 = df.Close.rolling(20).mean()

fig, ax = plt.subplots(figsize = (20,6))
ax.plot(df.Close)
plt.xlabel("Stock Price")
ax.plot(MA10, 'r', label='moving avg 10')
ax.plot(MA20, 'g', label='moving avg 20')
ax.legend()
plt.title("Moving Avg. of closing stock 10days, 20days ")
stm.pyplot(fig)

#daily returns line 
stm.subheader(f"Daily Return in percentage(%) of {select_stock}")
daily_return = df.Close.pct_change()
fig1 = plt.figure(figsize = (20,6))
daily_return.plot(legend=True, linestyle='dashed', marker='o')
plt.title("Daily returns line chart in percentage")
plt.xlabel("Daily-return (%)")
plt.ylabel("Days count")
stm.pyplot(fig1)
#histogram
stm.subheader("Number of counts a particulr returns")
fig2 = plt.figure(figsize=(20,6))
daily_return.hist(bins = 50)
plt.title("Daily returns count")
plt.xlabel("Daily return")
plt.ylabel("counts")
stm.pyplot(fig2)



#Prediction and model 
#train test split
train_data = pd.DataFrame(df['Close'])[0:int(len(df)*0.70)]
test_data = pd.DataFrame(df['Close'][int(len(df)*0.70):len(df)])

#scaling
scaler = MinMaxScaler(feature_range=(0,1))
train_data_array = scaler.fit_transform(train_data)

#model fitting
model = load_model('stock_analysis.h5')

#final 
past_150days = train_data.tail(150)
final_df = pd.concat([past_150days, train_data], ignore_index = True)

final_data = scaler.fit_transform(final_df)

X_test = []
y_test = []
for i in range(150, final_data.shape[0]):
  X_test.append(final_data[i-150: i])
  y_test.append(final_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

y_pred = model.predict(X_test)

y_test_2d = [y_test]
true_y_pred = scaler.inverse_transform(y_pred)
true_y_test = scaler.inverse_transform(y_test_2d)
true_y_test = true_y_test.reshape(len(true_y_pred), 1)

# scale_factor = scaler.scale_[0]
# y_pred = y_pred / scale_factor
# y_test = y_test / scale_factor




stm.subheader("Predict the closing price")
fig3 = plt.figure(figsize = (20,6))
plt.plot(true_y_test, 'g', label = 'Actual price')
plt.plot(true_y_pred, 'r', label = 'Predicted price')
plt.title("Actual vs Predict")
plt.ylabel("price")
plt.legend()
stm.pyplot(fig3)

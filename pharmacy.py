import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np

# Example: Reading the pharmacy_record DataFrame (adjust the path if necessary)
pharmacy_record = pd.read_csv('hospital_filtered.csv')

def preprocessing_for_training(df):
  partial_df = df[["Transaction DATE", "Transaction Quantity"]]
  partial_df["Transaction DATE"] = pd.to_datetime(partial_df["Transaction DATE"], format='%d-%m-%Y')
  partial_df["Transaction Quantity"] = pd.to_numeric(partial_df["Transaction Quantity"], errors='coerce')
  partial_df.dropna(inplace=True)
  partial_df.rename(columns={"Transaction DATE": 'ds',"Transaction Quantity":'y'}, inplace=True)
# #   partial_df.set_index("Transaction DATE", inplace=True)
  partial_df = partial_df.groupby("ds").agg("sum").reset_index()
#   partial_df = partial_df.asfreq('D')
# #   partial_df = partial_df.rename_axis('ds')
#   partial_df.dropna(inplace=True)
  return partial_df

def train_prophet_model(df):
  model = Prophet()
  model.fit(df)
  return model

def forecast_prophet(model, period, freq="D"):
  future_dates = model.make_future_dataframe(periods=period, freq=freq)
  forecast = model.predict(future_dates)
  return forecast

def pharmacy():
  # Add "None" as the initial selection option
  options = ["None"] + list(pharmacy_record['Description'])
  selected_category = st.sidebar.selectbox("Select an option", options)

  # Proceed only if a category other than "None" is selected
  if selected_category != "None":
      selected_item = pharmacy_record.loc[pharmacy_record['Description'] == selected_category, 'Item Code'].iloc[0]
      data = pd.read_csv(f'./{selected_item}.csv')
      
      if len(data) > 2:
          data = preprocessing_for_training(data)
          model = train_prophet_model(data)
          
          period = st.sidebar.selectbox("Select No of days to predict", list(range(1, 16)))
          forecast = forecast_prophet(model, period)
          forecast = forecast[['ds', 'yhat']]
          
          # Adjust the forecast values
          forecast['yhat'] = np.ceil(forecast['yhat'].abs()).clip(lower=1).astype(int)
          forecast = forecast.rename(columns={'ds': 'date', 'yhat': 'demand'})
          
          # Display the selected category and forecast data in the app
          st.markdown(f"<h3>{selected_category}</h3>", unsafe_allow_html=True)
          st.markdown(f"<div style='display: flex; justify-content: center;'>{forecast.tail(period).to_html(index=False)}</div>", unsafe_allow_html=True)
      else:
          st.write("Data contains less than 2 rows.")
  else:
      st.write("Please select an item to proceed.")



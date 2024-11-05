import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np
clinic_record = pd.read_csv('clinic_filtered.csv')
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

def clinic():
    selected_category = st.sidebar.selectbox("Select an option", list(clinic_record['Description']))
    selected_item = clinic_record.loc[clinic_record['Description'] == selected_category, 'Item Code'].iloc[0]
    data = pd.read_csv(f'./{selected_item}.csv')
    if len(data) > 2:
        data = preprocessing_for_training(data)
        model = train_prophet_model(data)
        period = st.sidebar.selectbox("Select No of days to predict", list(range(1, 16)))
        forcast = forecast_prophet(model, period)
        forcast = forcast[['ds', 'yhat']]
        forcast['yhat'] = np.ceil(forcast['yhat'].abs()).clip(lower=1).astype(int)
        forcast = forcast.rename(columns={'ds': 'date', 'yhat': 'demand'})
        st.markdown(f"<h3>{selected_category}</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='display: flex; justify-content: center;'>{forcast.tail(period).to_html(index=False)}</div>", unsafe_allow_html=True)
    else:
       print("Data contains less than 2 rows.")


from fastapi import FastAPI
import uvicorn
import joblib
import numpy as np
import pandas as pd

# Load the data
microinverter_df = pd.read_csv("2264002\\Microinverters\\202143051106_readings.csv", low_memory = False)
meter_df = pd.read_csv("2264002\\Meters\\202103021197eim1_readings.csv")
microinverter_model = joblib.load("microinverter_model.pkl")
microinverter_scaler = joblib.load("microinverter_scale.pkl")
microinverter_cutoff = joblib.load("microinverter_cutoff.pkl")
meter_model = joblib.load("meter_model.pkl")
meter_scaler = joblib.load("meter_scaler.pkl")
meter_cutoff = joblib.load("meter_cutoff.pkl")
microinverter_df['date'] = pd.to_datetime(microinverter_df["date"], utc = True)
microinverter_df['date'] = microinverter_df['date'] - pd.Timedelta(hours=8)
microinverter_df['date_only'] = microinverter_df["date"].dt.date
meter_df['date'] = pd.to_datetime(meter_df["date"], utc = True)
meter_df['date'] = meter_df['date'] - pd.Timedelta(hours=8)
meter_df['date_only'] = meter_df["date"].dt.date
meter_df1 = meter_df[meter_df['channel_id'] == 49812427]
meter_df2 = meter_df[meter_df['channel_id'] == 49812428]
meter_df2.set_index('date', inplace=True)
meter_df2.reset_index(inplace=True)


# Function to create sequences for the LSTM Model
def create_sequences(data, time_steps):
    sequences = []
    for i in range(len(data) - time_steps):
        seq = data[i:i + time_steps]
        sequences.append(seq)
    return np.array(sequences)

# Create the FastAPI instance
app = FastAPI()

@app.get("/")
def intro():
    return {"Message" : "ML Model"}

# Create a route to check for anomalies
@app.get("/predict/{date}")
def prediction(date: str):

    date = pd.to_datetime(date, utc = True).date()

    # Making a dataframe for the given date
    micro_start_index = microinverter_df[microinverter_df['date_only'] == date].index[0]
    microinverter_date = microinverter_df[microinverter_df['date_only'] == date]
    micro_prev_data = microinverter_df.iloc[micro_start_index-20: micro_start_index]
    microinverter_combined = pd.concat([micro_prev_data, microinverter_date])

    meter_start_index1 = meter_df1[meter_df1['date_only'] == date].index[0]
    meter_date1 = meter_df1[meter_df1['date_only'] == date]
    meter_prev_data1 = meter_df1.iloc[meter_start_index1-30: meter_start_index1]
    meter_combined1 = pd.concat([meter_prev_data1, meter_date1])

    meter_start_index2 = meter_df2[meter_df2['date_only'] == date].index[0]
    meter_date2 = meter_df2[meter_df2['date_only'] == date]
    meter_prev_data2 = meter_df2.iloc[meter_start_index2-30: meter_start_index2]
    meter_combined2 = pd.concat([meter_prev_data2, meter_date2])

    # Preprocessing the data
    microinverter_combined['minute'] = microinverter_combined['date'].dt.minute
    microinverter_combined['hour'] = microinverter_combined['date'].dt.hour
    microinverter_combined['day'] = microinverter_combined['date'].dt.day
    microinverter_combined['month'] = microinverter_combined['date'].dt.month
    microinverter_combined['energy_produced'] = microinverter_scaler.transform(microinverter_combined[['energy_produced']])
    microinverter_combined['energy_produced'] *=10
    
    # Running the model
    microinverter_features = microinverter_combined[['minute', 'hour', 'day', 'month', 'energy_produced']].values
    microinverter_X = create_sequences(microinverter_features, 20)
    microinverter_pred = microinverter_model.predict(microinverter_X)
    
    # Calculating the error and predicting anomalies
    microinverter_error = np.mean(np.square(microinverter_pred-microinverter_X), axis = 1)
    microinverter_final_df = pd.DataFrame(microinverter_combined[20:])
    microinverter_final_df['MSE'] = microinverter_error[:, 4]
    microinverter_final_df['cutoff'] = microinverter_cutoff
    microinverter_final_df['energy_produced'] /=10
    microinverter_final_df['energy_produced'] = microinverter_scaler.inverse_transform(microinverter_final_df[['energy_produced']])
    microinverter_final_df['anomaly'] = microinverter_final_df['MSE'] > microinverter_final_df['cutoff']


    # Preprocessing the data
    meter_combined2['curr_w'] = meter_scaler.transform(meter_combined2[['curr_w']])
    meter_features2 = meter_combined2[['curr_w']].values
    meter_X2 = create_sequences(meter_features2, 30)
    meter_pred2 = meter_model.predict(meter_X2)

    # Calculating the error and predicting anomalies
    meter_error2 = np.mean(np.square(meter_pred2-meter_X2), axis = 1)
    meter_final_df2 = pd.DataFrame(meter_combined2[30:])
    meter_final_df2['MSE'] = meter_error2
    meter_final_df2['cutoff'] = meter_cutoff
    meter_final_df2['curr_w'] = meter_scaler.inverse_transform(meter_final_df2[['curr_w']])
    meter_final_df2['anomaly'] = meter_final_df2['MSE'] > meter_final_df2['cutoff']

    # Preprocessing the data
    meter_combined1['curr_w'] = meter_scaler.transform(meter_combined1[['curr_w']])
    meter_features1 = meter_combined1[['curr_w']].values
    meter_X1 = create_sequences(meter_features1, 30)
    meter_pred1 = meter_model.predict(meter_X1)

    # Calculating the error and predicting anomalies
    meter_error1 = np.mean(np.square(meter_pred1-meter_X1), axis = 1)
    meter_final_df1 = pd.DataFrame(meter_combined1[30:])
    meter_final_df1['MSE'] = meter_error1
    meter_final_df1['cutoff'] = meter_cutoff
    meter_final_df1['curr_w'] = meter_scaler.inverse_transform(meter_final_df1[['curr_w']])
    meter_final_df1['anomaly'] = meter_final_df1['MSE'] > meter_final_df1['cutoff']

    microinverter_final_df = microinverter_final_df.replace([np.inf, -np.inf, np.nan], 0)
    meter_final_df1 = meter_final_df1.replace([np.inf, -np.inf, np.nan], 0)
    meter_final_df2 = meter_final_df2.replace([np.inf, -np.inf, np.nan], 0)

    # Returning the dataframes with anomalies
    return {"Microinverter Dataframe": microinverter_final_df.to_dict(orient='records'),
            "Meter Dataframe 1": meter_final_df1.to_dict(orient='records'),
            "Meter Dataframe 2": meter_final_df2.to_dict(orient='records')}
    

# Run the app
if __name__ == "__main__":
    uvicorn.run(app)
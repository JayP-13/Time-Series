import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the dataset into the code
df = pd.read_csv('daily_temperature.csv')

# Convert the Date column to datetime and sort data
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Prepare the dataset for prophet
df_prophet = df.rename(columns={'Date': 'ds', 'Tempature': 'y'})

# Selecting NY
df_prophet = df_prophet[df_prophet['City'] == 'New York']

# Check the dataframe
print(df_prophet.head())

# Initialize and fit the model
model = Prophet(daily_seasonality=True)
model.fit(df_prophet)

# Dataframe for predictions
future = model.make_future_dataframe(periods=0)
forecast = model.predict(future)

# Merge the forecast with the original data
df_prophet.set_index('ds', inplace=True)
forecast.set_index('ds', inplace=True)
df_merged = df_prophet.join(forecast[['yhat', 'yhat_lower', 'yhat_upper']], how='inner')

# Reset the index for plotting purposes
df_merged.reset_index(inplace=True)

# Plot the actual vs predicted temps
plt.figure(figsize=(10, 6))
plt.plot(df_merged['ds'], df_merged['y'], 'b-', label='Actual Temperature', marker='o', markersize=8)
plt.plot(df_merged['ds'], df_merged['yhat'], 'r-', label='Predicted Temperature', marker='o', markersize=8)

for i, txt in enumerate(df_merged['y']):
    plt.annotate(round(txt, 2), (df_merged['ds'][i], df_merged['y'][i]), textcoords="offset points", xytext=(0,10), ha='center')

for i, txt in enumerate(df_merged['yhat']):
    plt.annotate(round(txt, 2), (df_merged['ds'][i], df_merged['yhat'][i]), textcoords="offset points", xytext=(0,10), ha='center')

# Code to run the functions
plt.legend()
plt.xlabel('Date')
plt.ylabel('Tempature')
plt.title('Actual vs Predicted Temperatures')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Combining the datasets


```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Display the first few rows of each DataFrame as scrollable tables in Jupyter Notebook
from IPython.display import display
from tabulate import tabulate
import holidays

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')
```

#### Types of information
Now that we've examined both the energy data and the forecasting data, there's only one thing left to do: combine them. What we aim for is a final dataset that can be integrated into various other datasets with minimal changes. This dataset should contain three types of information (possibly four, but more on that later):

1. Energy consumption data
2. Temperature forecasting data
3. Information about the day and whether it's a holiday

Furthermore, these data sets need to be synchronized in terms of time. The energy data is collected hourly, while the temperature data is gathered every six hours. Both will be aggregated to daily data, but in different ways. The temperature will be averaged throughout the day—though we might consider using the sum, as this would eliminate negative temperatures, which could be interesting to explore. The energy data will be aggregated by summing the data.

#### Missing data issue
Then there's the issue of missing data, which is why we might need a fourth type of information. One approach is to use linear interpolation to fill in the gaps. While this is a reliable method for handling missing data, it does introduce some uncertainty. If there are long periods of missing data, this could significantly affect the variance of the models. Another approach is to use a flagging method, where we add a variable that flags dates with missing information as "1" and days with complete data as "0". This could be particularly useful for methods with coefficients, like OLS, as well as for methods like Random Forest Regression. In the latter case, the flag would help the model predict a "0" on days where we don't know the actual consumption, thereby providing success metrics based solely on the model's ability to predict known data. We will create datasets for both scenarios.

Let's get started.

## Energy data

### Loading Energy data


```python
# Load the electricity consumption dataset
consumption_filepath = 'C:/Users/madsh/OneDrive/Dokumenter/kandidat/Fællesmappe/Speciale/Forecasting-energy-consumption-in-Denmark/Data/Energy/Production and Consumption - Settlement.csv'
consumption_df = pd.read_csv(consumption_filepath)

# Display the electricity consumption DataFrame
print("Electricity Consumption Data:")
display(consumption_df.head())
```

    Electricity Consumption Data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HourUTC</th>
      <th>HourDK</th>
      <th>PriceArea</th>
      <th>CentralPowerMWh</th>
      <th>LocalPowerMWh</th>
      <th>CommercialPowerMWh</th>
      <th>LocalPowerSelfConMWh</th>
      <th>OffshoreWindLt100MW_MWh</th>
      <th>OffshoreWindGe100MW_MWh</th>
      <th>OnshoreWindLt50kW_MWh</th>
      <th>...</th>
      <th>ExchangeNO_MWh</th>
      <th>ExchangeSE_MWh</th>
      <th>ExchangeGE_MWh</th>
      <th>ExchangeNL_MWh</th>
      <th>ExchangeGreatBelt_MWh</th>
      <th>GrossConsumptionMWh</th>
      <th>GridLossTransmissionMWh</th>
      <th>GridLossInterconnectorsMWh</th>
      <th>GridLossDistributionMWh</th>
      <th>PowerToHeatMWh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-03-25T22:00:00</td>
      <td>2005-03-25T23:00:00</td>
      <td>DK1</td>
      <td>917.400024</td>
      <td>760.206787</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>4.857113</td>
      <td>0.000000</td>
      <td>0.022235</td>
      <td>...</td>
      <td>496.000000</td>
      <td>-97.099998</td>
      <td>-297.700012</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1842.515015</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-03-25T22:00:00</td>
      <td>2005-03-25T23:00:00</td>
      <td>DK2</td>
      <td>920.900024</td>
      <td>271.390656</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3.329244</td>
      <td>2.978000</td>
      <td>0.006352</td>
      <td>...</td>
      <td>NaN</td>
      <td>386.600006</td>
      <td>-251.000000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1386.037964</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-03-25T21:00:00</td>
      <td>2005-03-25T22:00:00</td>
      <td>DK1</td>
      <td>1079.099976</td>
      <td>772.546753</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>4.533219</td>
      <td>0.000000</td>
      <td>0.015721</td>
      <td>...</td>
      <td>808.599976</td>
      <td>169.699997</td>
      <td>-870.299988</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2010.311157</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-03-25T21:00:00</td>
      <td>2005-03-25T22:00:00</td>
      <td>DK2</td>
      <td>908.099976</td>
      <td>296.979065</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>5.279539</td>
      <td>44.057999</td>
      <td>0.008078</td>
      <td>...</td>
      <td>NaN</td>
      <td>631.500000</td>
      <td>-447.000000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1501.229004</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-03-25T20:00:00</td>
      <td>2005-03-25T21:00:00</td>
      <td>DK1</td>
      <td>1125.400024</td>
      <td>833.091309</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3.346528</td>
      <td>0.000000</td>
      <td>0.012367</td>
      <td>...</td>
      <td>991.400024</td>
      <td>431.600006</td>
      <td>-1245.199951</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2180.373779</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>


#### Electricity Consumption Data
* `HourUTC` and `HourDK`: Timestamps in UTC and Danish time
* `PriceArea`: Price area (either DK1 or DK2)
* `CentralPowerMWh`, `LocalPowerMWh`, etc.: Various measures of electricity consumption and production
* `GrossConsumptionMWh`: The measure we are interested in for electricity consumption


```python
# Convert the relevant columns to datetime format
consumption_df['HourDK'] = pd.to_datetime(consumption_df['HourDK'])

# Summing up GrossConsumptionMWh for both DK1 and DK2 for each time slot
consumption_grouped_df = consumption_df.groupby('HourDK')['GrossConsumptionMWh'].sum().reset_index()

# Show the first few rows of the grouped DataFrame
consumption_grouped_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HourDK</th>
      <th>GrossConsumptionMWh</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-01 00:00:00</td>
      <td>3370.256592</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-01 01:00:00</td>
      <td>3237.832763</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-01 02:00:00</td>
      <td>3101.580811</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-01 03:00:00</td>
      <td>2963.392211</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-01 04:00:00</td>
      <td>2854.805420</td>
    </tr>
  </tbody>
</table>
</div>



### Linear interpolation


```python
# Generate a complete date range
complete_date_range = pd.date_range(start=consumption_grouped_df['HourDK'][0], end=consumption_grouped_df['HourDK'][len(consumption_grouped_df['HourDK'])-1], freq='H')

# Put HourDK as DataFrame index
consumption_grouped_df.set_index('HourDK', inplace=True)

# Reindex the DataFrame to include all dates and set NaN for missing dates
consumption_grouped_df = consumption_grouped_df.reindex(complete_date_range)
```


```python
# Reset index to make 'HourDK' a column again
line_numbers = consumption_grouped_df.index[consumption_grouped_df['GrossConsumptionMWh'].isna()].tolist()
selected_rows = consumption_grouped_df.loc[line_numbers]
print(selected_rows)
missing_before = consumption_grouped_df['GrossConsumptionMWh'].isna().sum()
print(f"\nNumber of missing values before interpolation: {missing_before}")
```

                         GrossConsumptionMWh
    2005-03-26 00:00:00                  NaN
    2005-03-26 01:00:00                  NaN
    2005-03-26 02:00:00                  NaN
    2005-03-26 03:00:00                  NaN
    2005-03-26 04:00:00                  NaN
    ...                                  ...
    2023-04-09 19:00:00                  NaN
    2023-04-09 20:00:00                  NaN
    2023-04-09 21:00:00                  NaN
    2023-04-09 22:00:00                  NaN
    2023-04-09 23:00:00                  NaN
    
    [1531 rows x 1 columns]
    
    Number of missing values before interpolation: 1531
    


```python
# Perform linear interpolation
combined_daily_interpolation_df = consumption_grouped_df.interpolate(method='linear')

line_numbers = consumption_grouped_df.index[consumption_grouped_df['GrossConsumptionMWh'].isna()].tolist()
selected_rows = combined_daily_interpolation_df.loc[line_numbers]
print(selected_rows)

# Count and print the number of missing values after interpolation
missing_after = combined_daily_interpolation_df['GrossConsumptionMWh'].isna().sum()
print(f"\nNumber of missing values after interpolation: {missing_after}")
```

                         GrossConsumptionMWh
    2005-03-26 00:00:00          3219.117071
    2005-03-26 01:00:00          3209.681163
    2005-03-26 02:00:00          3200.245254
    2005-03-26 03:00:00          3190.809346
    2005-03-26 04:00:00          3181.373438
    ...                                  ...
    2023-04-09 19:00:00          2976.840017
    2023-04-09 20:00:00          2974.051890
    2023-04-09 21:00:00          2971.263764
    2023-04-09 22:00:00          2968.475637
    2023-04-09 23:00:00          2965.687510
    
    [1531 rows x 1 columns]
    
    Number of missing values after interpolation: 0
    


```python
# Reset index to make 'HourDK' a column again
combined_daily_interpolation_df.reset_index(inplace=True)
combined_daily_interpolation_df.rename(columns={'index': 'HourDK'}, inplace=True)

# Print the DataFrame to check the results
print(combined_daily_interpolation_df)
```

                        HourDK  GrossConsumptionMWh
    0      2005-01-01 00:00:00          3370.256592
    1      2005-01-01 01:00:00          3237.832763
    2      2005-01-01 02:00:00          3101.580811
    3      2005-01-01 03:00:00          2963.392211
    4      2005-01-01 04:00:00          2854.805420
    ...                    ...                  ...
    161371 2023-05-30 19:00:00          3935.964505
    161372 2023-05-30 20:00:00          3764.163099
    161373 2023-05-30 21:00:00          3655.639568
    161374 2023-05-30 22:00:00          3663.715933
    161375 2023-05-30 23:00:00          3308.564927
    
    [161376 rows x 2 columns]
    

### Flagging


```python
consumption_df = pd.read_csv(consumption_filepath)

# Convert the relevant columns to datetime format
consumption_df['HourDK'] = pd.to_datetime(consumption_df['HourDK'])

# Summing up GrossConsumptionMWh for both DK1 and DK2 for each time slot
consumption_grouped_df['HourDK'] = pd.to_datetime(consumption_df['HourDK'])
consumption_grouped_flagged_df = consumption_df.groupby('HourDK')['GrossConsumptionMWh'].sum().reset_index()

# Generate a complete date range
complete_date_range = pd.date_range(start=consumption_grouped_flagged_df['HourDK'].min(), end=consumption_grouped_flagged_df['HourDK'].max(), freq='H')

# Put HourDK as DataFrame index
consumption_grouped_flagged_df.set_index('HourDK', inplace=True)

# Reindex the DataFrame to include all dates and set NaN for missing dates
consumption_grouped_flagged_df = consumption_grouped_flagged_df.reindex(complete_date_range)

# Check the number of missing values before flagging
missing_before = consumption_grouped_flagged_df['GrossConsumptionMWh'].isna().sum()
print(f"Number of missing values before flagging: {missing_before}")

# Create the 'flagged' column
consumption_grouped_flagged_df['flagged'] = np.where(
    (consumption_grouped_flagged_df['GrossConsumptionMWh'] < 1) | 
    consumption_grouped_flagged_df['GrossConsumptionMWh'].isna(), 
    1, 
    0
)

# Reset the index to make HourDK a column again and remove the duplicate index
consumption_grouped_flagged_df.reset_index(inplace=True)
consumption_grouped_flagged_df.rename(columns={'index': 'HourDK'}, inplace=True)

# Show the resulting DataFrame
print(consumption_grouped_flagged_df)

# Count and print the number of missing values after flagging
missing_after = consumption_grouped_flagged_df['GrossConsumptionMWh'].isna().sum()
print(f"\nNumber of missing values after Flagging: {missing_after}")
print(f"Number of 'Flagged' observations: {sum(consumption_grouped_flagged_df['flagged'])}")
```

    Number of missing values before flagging: 1531
                        HourDK  GrossConsumptionMWh  flagged
    0      2005-01-01 00:00:00          3370.256592        0
    1      2005-01-01 01:00:00          3237.832763        0
    2      2005-01-01 02:00:00          3101.580811        0
    3      2005-01-01 03:00:00          2963.392211        0
    4      2005-01-01 04:00:00          2854.805420        0
    ...                    ...                  ...      ...
    161371 2023-05-30 19:00:00          3935.964505        0
    161372 2023-05-30 20:00:00          3764.163099        0
    161373 2023-05-30 21:00:00          3655.639568        0
    161374 2023-05-30 22:00:00          3663.715933        0
    161375 2023-05-30 23:00:00          3308.564927        0
    
    [161376 rows x 3 columns]
    
    Number of missing values after Flagging: 1531
    Number of 'Flagged' observations: 1531
    

### Aggregating to daily


```python
# Resample the interpolated electricity consumption data to daily level and sum up GrossConsumptionMWh
combined_daily_interpolation_df = combined_daily_interpolation_df.resample('D', on='HourDK').sum().reset_index()

# Show the first few rows of the daily aggregated DataFrame
print("Interpolated energy data:")
print(combined_daily_interpolation_df)

# Resample the interpolated electricity consumption data to daily level and sum up GrossConsumptionMWh
consumption_daily_flagged_df = consumption_grouped_flagged_df.resample('D', on='HourDK').agg({'GrossConsumptionMWh': 'sum', 'flagged': 'max'}).reset_index()
consumption_daily_flagged_df['GrossConsumptionMWh'] = np.where(consumption_daily_flagged_df['flagged'] == 1, 0, consumption_daily_flagged_df['GrossConsumptionMWh'])
# Show the first few rows of the daily aggregated DataFrame
print("\nFlagged energy data:")
print(consumption_daily_flagged_df)
```

    Interpolated energy data:
             HourDK  GrossConsumptionMWh
    0    2005-01-01         84760.194094
    1    2005-01-02         91208.524416
    2    2005-01-03        112086.718383
    3    2005-01-04        114699.218872
    4    2005-01-05        113435.680422
    ...         ...                  ...
    6719 2023-05-26         91966.741455
    6720 2023-05-27         79738.191501
    6721 2023-05-28         80406.440116
    6722 2023-05-29         82766.586296
    6723 2023-05-30         89449.965396
    
    [6724 rows x 2 columns]
    
    Flagged energy data:
             HourDK  GrossConsumptionMWh  flagged
    0    2005-01-01         84760.194094        0
    1    2005-01-02         91208.524416        0
    2    2005-01-03        112086.718383        0
    3    2005-01-04        114699.218872        0
    4    2005-01-05        113435.680422        0
    ...         ...                  ...      ...
    6719 2023-05-26         91966.741455        0
    6720 2023-05-27         79738.191501        0
    6721 2023-05-28         80406.440116        0
    6722 2023-05-29         82766.586296        0
    6723 2023-05-30         89449.965396        0
    
    [6724 rows x 3 columns]
    

Now that the energy data has been prepared for integration into a combined dataset, let's now prepare the weather forecast data.

## Weather Forecast Data

### Loading Energy data


```python
# Load the weather forecast dataset
weather_filepath = 'C:/Users/madsh/OneDrive/Dokumenter/kandidat/Fællesmappe/Speciale/Forecasting-energy-consumption-in-Denmark/Data/Weather forecasts/combined_forecasts_2005-2023.csv'
weather_df = pd.read_csv(weather_filepath)

# Display the weather forecast DataFrame
print("Weather Forecast Data:")
display(weather_df.head())
```

    Weather Forecast Data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>valid_time</th>
      <th>time</th>
      <th>step</th>
      <th>t2m</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-01 06:00:00</td>
      <td>2005-01-01</td>
      <td>0 days 06:00:00</td>
      <td>3.090616</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-01 12:00:00</td>
      <td>2005-01-01</td>
      <td>0 days 12:00:00</td>
      <td>4.884995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-01 18:00:00</td>
      <td>2005-01-01</td>
      <td>0 days 18:00:00</td>
      <td>5.163322</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-02 00:00:00</td>
      <td>2005-01-01</td>
      <td>1 days 00:00:00</td>
      <td>5.461974</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-02 06:00:00</td>
      <td>2005-01-01</td>
      <td>1 days 06:00:00</td>
      <td>3.571397</td>
    </tr>
  </tbody>
</table>
</div>


#### Weather Data
* `valid_time`: The valid time for the weather forecast
* `time`: The date of the forecast
* `step`: The step length for the forecast (e.g., "0 days 06:00:00" means 6 hours ahead)
* `t2m`: Temperature in degrees Celsius

### Aggregating to Daily


```python
weather_df['valid_time'] = pd.to_datetime(weather_df['valid_time'])

# Resample the weather data to daily level and average the temperature (t2m)
weather_daily_df = weather_df.resample('D', on='valid_time').mean().reset_index()

# Show the first few rows of the daily aggregated DataFrame
weather_daily_df.head()

# Filter out the 00:00 observations for each day to keep the 'step' information
weather_step_df = weather_df[weather_df['valid_time'].dt.hour == 0]

# Merge the filtered 'step' DataFrame with the daily averaged temperature DataFrame on 'valid_time'
weather_daily_with_step_df = pd.merge(weather_daily_df, weather_step_df[['valid_time', 'step']], on='valid_time', how='left')

# Set the 'step' for the first day of the year to be "0 days"
weather_daily_with_step_df.loc[0, 'step'] = pd.Timedelta(days=0)
weather_daily_with_step_df['step'].fillna(pd.Timedelta('0 days 00:00:00'), inplace=True)


# Show the first few rows of the daily aggregated DataFrame with 'step'
weather_daily_with_step_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>valid_time</th>
      <th>t2m</th>
      <th>step</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-01</td>
      <td>4.379644</td>
      <td>0 days 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-02</td>
      <td>3.912904</td>
      <td>1 days 00:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-03</td>
      <td>4.320021</td>
      <td>2 days 00:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-04</td>
      <td>6.146450</td>
      <td>3 days 00:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-05</td>
      <td>5.212295</td>
      <td>4 days 00:00:00</td>
    </tr>
  </tbody>
</table>
</div>



#### Convert step to numerical:


```python
# First, convert 'step' to string
weather_daily_with_step_df['step'] = weather_daily_with_step_df['step'].astype(str)

# Extract the number of days and convert to float first (this will convert 'nan' to NaN)
weather_daily_with_step_df['step_days'] = weather_daily_with_step_df['step'].str.split(' ').str[0].astype(float)

# Now filter out the rows where 'step_days' is NaN if you want
weather_daily_with_step_df = weather_daily_with_step_df[weather_daily_with_step_df['step_days'].notna()]

# Finally, convert to integer
weather_daily_with_step_df['step_days'] = weather_daily_with_step_df['step_days'].astype(int)

# Drop the original 'step' column
weather_daily_with_step_df.drop('step', axis=1, inplace=True)

weather_daily_with_step_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>valid_time</th>
      <th>t2m</th>
      <th>step_days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-01</td>
      <td>4.379644</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-02</td>
      <td>3.912904</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-03</td>
      <td>4.320021</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-04</td>
      <td>6.146450</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-05</td>
      <td>5.212295</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have prepared the weather forecasting data, we can combine the two datasets. Again, we will create two datasets: one for interpolation and one for flagging.

## Combining


```python
# Merge the daily aggregated electricity consumption data with the daily aggregated weather data
combined_daily_interpolation_df = pd.merge(combined_daily_interpolation_df, weather_daily_with_step_df, 
                             left_on='HourDK', right_on='valid_time', how='inner')

# Drop the redundant 'valid_time' column
combined_daily_interpolation_df.drop('valid_time', axis=1, inplace=True)

# Show the first few rows of the combined DataFrame
print("Interpolated combined data:")
display(combined_daily_interpolation_df.head())

# Merge the daily aggregated electricity consumption data with the daily aggregated weather data
combined_daily_flagged_df = pd.merge(consumption_daily_flagged_df, weather_daily_with_step_df, 
                             left_on='HourDK', right_on='valid_time', how='inner')

# Drop the redundant 'valid_time' column
combined_daily_flagged_df.drop('valid_time', axis=1, inplace=True)

# Show the first few rows of the combined DataFrame
print("\nFlagged combined data:")
display(combined_daily_flagged_df.head())
```

    Interpolated combined data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HourDK</th>
      <th>GrossConsumptionMWh</th>
      <th>t2m</th>
      <th>step_days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-01</td>
      <td>84760.194094</td>
      <td>4.379644</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-02</td>
      <td>91208.524416</td>
      <td>3.912904</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-03</td>
      <td>112086.718383</td>
      <td>4.320021</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-04</td>
      <td>114699.218872</td>
      <td>6.146450</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-05</td>
      <td>113435.680422</td>
      <td>5.212295</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


    
    Flagged combined data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HourDK</th>
      <th>GrossConsumptionMWh</th>
      <th>flagged</th>
      <th>t2m</th>
      <th>step_days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-01</td>
      <td>84760.194094</td>
      <td>0</td>
      <td>4.379644</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-02</td>
      <td>91208.524416</td>
      <td>0</td>
      <td>3.912904</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-03</td>
      <td>112086.718383</td>
      <td>0</td>
      <td>4.320021</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-04</td>
      <td>114699.218872</td>
      <td>0</td>
      <td>6.146450</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-05</td>
      <td>113435.680422</td>
      <td>0</td>
      <td>5.212295</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


Now that we have combined the datasets, it is time to add the information about what day it is.

## Adding additional variable for charateristics for the days and holidays

To enrich the dataset, we'll add the following features:

1. **Day of the Week**: A categorical variable representing which day of the week a given date falls on.
2. **Month of the Year**: A categorical variable representing the month in which a given date falls.
3. **Holiday Status**: A binary variable indicating whether a given date is a public holiday in Denmark or not.


```python
# Add 'Day_of_Week' and 'Month_of_Year' columns
combined_daily_interpolation_df['Day_of_Week'] = combined_daily_interpolation_df['HourDK'].dt.day_name()
combined_daily_interpolation_df['Month_of_Year'] = combined_daily_interpolation_df['HourDK'].dt.month_name()

# Show the first few rows of the combined DataFrame
print("Interpolated combined data:")
display(combined_daily_interpolation_df.head())

# Add 'Day_of_Week' and 'Month_of_Year' columns
combined_daily_flagged_df['Day_of_Week'] = combined_daily_flagged_df['HourDK'].dt.day_name()
combined_daily_flagged_df['Month_of_Year'] = combined_daily_flagged_df['HourDK'].dt.month_name()

# Show the first few rows to confirm the added columns
print("\nFlagged combined data:")
display(combined_daily_flagged_df.head())
```

    Interpolated combined data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HourDK</th>
      <th>GrossConsumptionMWh</th>
      <th>t2m</th>
      <th>step_days</th>
      <th>Day_of_Week</th>
      <th>Month_of_Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-01</td>
      <td>84760.194094</td>
      <td>4.379644</td>
      <td>0</td>
      <td>Saturday</td>
      <td>January</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-02</td>
      <td>91208.524416</td>
      <td>3.912904</td>
      <td>1</td>
      <td>Sunday</td>
      <td>January</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-03</td>
      <td>112086.718383</td>
      <td>4.320021</td>
      <td>2</td>
      <td>Monday</td>
      <td>January</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-04</td>
      <td>114699.218872</td>
      <td>6.146450</td>
      <td>3</td>
      <td>Tuesday</td>
      <td>January</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-05</td>
      <td>113435.680422</td>
      <td>5.212295</td>
      <td>4</td>
      <td>Wednesday</td>
      <td>January</td>
    </tr>
  </tbody>
</table>
</div>


    
    Flagged combined data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HourDK</th>
      <th>GrossConsumptionMWh</th>
      <th>flagged</th>
      <th>t2m</th>
      <th>step_days</th>
      <th>Day_of_Week</th>
      <th>Month_of_Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-01</td>
      <td>84760.194094</td>
      <td>0</td>
      <td>4.379644</td>
      <td>0</td>
      <td>Saturday</td>
      <td>January</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-02</td>
      <td>91208.524416</td>
      <td>0</td>
      <td>3.912904</td>
      <td>1</td>
      <td>Sunday</td>
      <td>January</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-03</td>
      <td>112086.718383</td>
      <td>0</td>
      <td>4.320021</td>
      <td>2</td>
      <td>Monday</td>
      <td>January</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-04</td>
      <td>114699.218872</td>
      <td>0</td>
      <td>6.146450</td>
      <td>3</td>
      <td>Tuesday</td>
      <td>January</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-05</td>
      <td>113435.680422</td>
      <td>0</td>
      <td>5.212295</td>
      <td>4</td>
      <td>Wednesday</td>
      <td>January</td>
    </tr>
  </tbody>
</table>
</div>


### Holidays


```python
# Generate the list of holidays for Denmark for the years 2005-2023
danish_holidays = [date for year in range(2005, 2024) for date, _ in holidays.Denmark(years=year).items()]
# Convert the list to a pandas Series and make sure it's in datetime format
danish_holidays = pd.Series(pd.to_datetime(danish_holidays))
# Convert the holiday dates to datetime format
danish_holidays = pd.to_datetime(danish_holidays)

# Create a new column 'Is_Holiday' and set it to 1 if the date is a holiday, 0 otherwise
combined_daily_interpolation_df['Is_Holiday'] = combined_daily_interpolation_df['HourDK'].isin(danish_holidays).astype(int)
combined_daily_flagged_df['Is_Holiday'] = combined_daily_flagged_df['HourDK'].isin(danish_holidays).astype(int)

# Show the first few rows of the combined DataFrame
print("Interpolated combined data:")
display(combined_daily_interpolation_df.head())

# Show the first few rows to confirm the added columns
print("\nFlagged combined data:")
display(combined_daily_flagged_df.head())
```

    Interpolated combined data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HourDK</th>
      <th>GrossConsumptionMWh</th>
      <th>t2m</th>
      <th>step_days</th>
      <th>Day_of_Week</th>
      <th>Month_of_Year</th>
      <th>Is_Holiday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-01</td>
      <td>84760.194094</td>
      <td>4.379644</td>
      <td>0</td>
      <td>Saturday</td>
      <td>January</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-02</td>
      <td>91208.524416</td>
      <td>3.912904</td>
      <td>1</td>
      <td>Sunday</td>
      <td>January</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-03</td>
      <td>112086.718383</td>
      <td>4.320021</td>
      <td>2</td>
      <td>Monday</td>
      <td>January</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-04</td>
      <td>114699.218872</td>
      <td>6.146450</td>
      <td>3</td>
      <td>Tuesday</td>
      <td>January</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-05</td>
      <td>113435.680422</td>
      <td>5.212295</td>
      <td>4</td>
      <td>Wednesday</td>
      <td>January</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    Flagged combined data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HourDK</th>
      <th>GrossConsumptionMWh</th>
      <th>flagged</th>
      <th>t2m</th>
      <th>step_days</th>
      <th>Day_of_Week</th>
      <th>Month_of_Year</th>
      <th>Is_Holiday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-01</td>
      <td>84760.194094</td>
      <td>0</td>
      <td>4.379644</td>
      <td>0</td>
      <td>Saturday</td>
      <td>January</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-02</td>
      <td>91208.524416</td>
      <td>0</td>
      <td>3.912904</td>
      <td>1</td>
      <td>Sunday</td>
      <td>January</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-03</td>
      <td>112086.718383</td>
      <td>0</td>
      <td>4.320021</td>
      <td>2</td>
      <td>Monday</td>
      <td>January</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-04</td>
      <td>114699.218872</td>
      <td>0</td>
      <td>6.146450</td>
      <td>3</td>
      <td>Tuesday</td>
      <td>January</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-05</td>
      <td>113435.680422</td>
      <td>0</td>
      <td>5.212295</td>
      <td>4</td>
      <td>Wednesday</td>
      <td>January</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


Now we are extremely close to have a final dataset. but there are a couple of steps left before we can save these dataframes as csv files and go nuts with putting the data into machine learning models. 

First When working with machine learning models for forecasting, using dummy variables instead of categorical variables can offer several advantages. Here's why:

#### Interpretability and Consistency Across Models:
Different machine learning algorithms handle categorical variables in different ways. Some might not even accept categorical variables as input, requiring pre-processing. Transforming categorical variables into a series of dummy variables (often called "one-hot encoding") can make it easier to compare and interpret results across different models.

#### Non-linear Relationships:
Using dummy variables allows the model to understand non-linear relationships between the categories and the dependent variable, which might not be easily captured if the categorical variables are left as-is or labeled in a numerical but ordinal way.

#### Interaction Effects:
Dummy variables make it easier to include interaction terms in your model. If you believe that the effect of one variable depends on the level of another variable, dummy variables make it straightforward to model these interactions.

#### Model Performance:
Last but not least, using dummy variables can actually improve model performance. When a categorical variable is converted to dummy variables, the model has more specific variables to "play" with, often resulting in a more accurate and robust model.

However, it's worth mentioning that introducing many dummy variables can also lead to the "curse of dimensionality," where the feature space becomes too large, thereby requiring more data for the model to generalize well. But this is often a manageable challenge, and techniques like dimensionality reduction can help if it becomes a problem.

So for these reasons, using dummy variables is often preferable to using categorical variables when forecasting with machine learning models. so lets convert the varibels:


```python
# Generate dummy variables for 'Day_of_Week' and 'Month_of_Year'
day_of_week_dummies = pd.get_dummies(combined_daily_interpolation_df['Day_of_Week'], prefix='Day')
month_of_year_dummies = pd.get_dummies(combined_daily_interpolation_df['Month_of_Year'], prefix='Month')

# Concatenate the original DataFrame with the dummy variables
combined_daily_interpolation_df = pd.concat([combined_daily_interpolation_df, day_of_week_dummies, month_of_year_dummies], axis=1)
combined_daily_flagged_df       = pd.concat([combined_daily_flagged_df,       day_of_week_dummies, month_of_year_dummies], axis=1)

# Drop the original 'Day_of_Week' and 'Month_of_Year' columns
combined_daily_interpolation_df.drop(['Day_of_Week', 'Month_of_Year'], axis=1, inplace=True)
combined_daily_flagged_df.drop(['Day_of_Week', 'Month_of_Year']     , axis=1, inplace=True)

# Define the desired column order
day_columns = ['Day_Monday', 'Day_Tuesday', 'Day_Wednesday', 'Day_Thursday', 'Day_Friday', 'Day_Saturday', 'Day_Sunday']
month_columns = ['Month_January', 'Month_February', 'Month_March', 'Month_April', 'Month_May', 'Month_June', 'Month_July', 'Month_August', 'Month_September', 'Month_October', 'Month_November', 'Month_December']
other_columns_inter = ['HourDK', 'GrossConsumptionMWh', 't2m', 'step_days','Is_Holiday']
other_columns_flag  = ['HourDK', 'GrossConsumptionMWh','flagged', 't2m', 'step_days','Is_Holiday']

# Combine all columns in the desired order
ordered_columns_inter = other_columns_inter + day_columns + month_columns
ordered_columns_flag = other_columns_flag + day_columns + month_columns

# Reorder the columns in the DataFrame
combined_daily_interpolation_df = combined_daily_interpolation_df[ordered_columns_inter]
combined_daily_flagged_df       = combined_daily_flagged_df[ordered_columns_flag]

# Show the first few rows to confirm the new column order
print("Interpolated combined data:")
display(combined_daily_interpolation_df.head())

# Show the first few rows to confirm the added columns
print("\nFlagged combined data:")
display(combined_daily_flagged_df.head())
```

    Interpolated combined data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HourDK</th>
      <th>GrossConsumptionMWh</th>
      <th>t2m</th>
      <th>step_days</th>
      <th>Is_Holiday</th>
      <th>Day_Monday</th>
      <th>Day_Tuesday</th>
      <th>Day_Wednesday</th>
      <th>Day_Thursday</th>
      <th>Day_Friday</th>
      <th>...</th>
      <th>Month_March</th>
      <th>Month_April</th>
      <th>Month_May</th>
      <th>Month_June</th>
      <th>Month_July</th>
      <th>Month_August</th>
      <th>Month_September</th>
      <th>Month_October</th>
      <th>Month_November</th>
      <th>Month_December</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-01</td>
      <td>84760.194094</td>
      <td>4.379644</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-02</td>
      <td>91208.524416</td>
      <td>3.912904</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-03</td>
      <td>112086.718383</td>
      <td>4.320021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-04</td>
      <td>114699.218872</td>
      <td>6.146450</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-05</td>
      <td>113435.680422</td>
      <td>5.212295</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>


    
    Flagged combined data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HourDK</th>
      <th>GrossConsumptionMWh</th>
      <th>flagged</th>
      <th>t2m</th>
      <th>step_days</th>
      <th>Is_Holiday</th>
      <th>Day_Monday</th>
      <th>Day_Tuesday</th>
      <th>Day_Wednesday</th>
      <th>Day_Thursday</th>
      <th>...</th>
      <th>Month_March</th>
      <th>Month_April</th>
      <th>Month_May</th>
      <th>Month_June</th>
      <th>Month_July</th>
      <th>Month_August</th>
      <th>Month_September</th>
      <th>Month_October</th>
      <th>Month_November</th>
      <th>Month_December</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-01</td>
      <td>84760.194094</td>
      <td>0</td>
      <td>4.379644</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-02</td>
      <td>91208.524416</td>
      <td>0</td>
      <td>3.912904</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-03</td>
      <td>112086.718383</td>
      <td>0</td>
      <td>4.320021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-04</td>
      <td>114699.218872</td>
      <td>0</td>
      <td>6.146450</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-05</td>
      <td>113435.680422</td>
      <td>0</td>
      <td>5.212295</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>


Now there is only one thing left to do: we need to remove all observations from 2023. This is because we are forecasting six months ahead, and the specific day from which we forecast is important. In that spirit, we need the last six months of the dataset to serve as our test set, which means it should end on December 31, 2022.


```python
combined_daily_interpolation_df = combined_daily_interpolation_df[combined_daily_interpolation_df['HourDK'] <= '2022-12-31']
combined_daily_flagged_df = combined_daily_flagged_df[combined_daily_flagged_df['HourDK'] <= '2022-12-31']

# Show the first few rows to confirm the new column order
print("Interpolated combined data:")
display(combined_daily_interpolation_df.tail())

# Show the first few rows to confirm the added columns
print("\nFlagged combined data:")
display(combined_daily_flagged_df.tail())
```

    Interpolated combined data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HourDK</th>
      <th>GrossConsumptionMWh</th>
      <th>t2m</th>
      <th>step_days</th>
      <th>Is_Holiday</th>
      <th>Day_Monday</th>
      <th>Day_Tuesday</th>
      <th>Day_Wednesday</th>
      <th>Day_Thursday</th>
      <th>Day_Friday</th>
      <th>...</th>
      <th>Month_March</th>
      <th>Month_April</th>
      <th>Month_May</th>
      <th>Month_June</th>
      <th>Month_July</th>
      <th>Month_August</th>
      <th>Month_September</th>
      <th>Month_October</th>
      <th>Month_November</th>
      <th>Month_December</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6569</th>
      <td>2022-12-27</td>
      <td>100264.310792</td>
      <td>2.246549</td>
      <td>179</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6570</th>
      <td>2022-12-28</td>
      <td>106942.629760</td>
      <td>2.202570</td>
      <td>180</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6571</th>
      <td>2022-12-29</td>
      <td>108750.475221</td>
      <td>2.503690</td>
      <td>181</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6572</th>
      <td>2022-12-30</td>
      <td>108998.128298</td>
      <td>2.414602</td>
      <td>182</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6573</th>
      <td>2022-12-31</td>
      <td>100746.457400</td>
      <td>2.286391</td>
      <td>183</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>


    
    Flagged combined data:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HourDK</th>
      <th>GrossConsumptionMWh</th>
      <th>flagged</th>
      <th>t2m</th>
      <th>step_days</th>
      <th>Is_Holiday</th>
      <th>Day_Monday</th>
      <th>Day_Tuesday</th>
      <th>Day_Wednesday</th>
      <th>Day_Thursday</th>
      <th>...</th>
      <th>Month_March</th>
      <th>Month_April</th>
      <th>Month_May</th>
      <th>Month_June</th>
      <th>Month_July</th>
      <th>Month_August</th>
      <th>Month_September</th>
      <th>Month_October</th>
      <th>Month_November</th>
      <th>Month_December</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6569</th>
      <td>2022-12-27</td>
      <td>100264.310792</td>
      <td>0</td>
      <td>2.246549</td>
      <td>179</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6570</th>
      <td>2022-12-28</td>
      <td>106942.629760</td>
      <td>0</td>
      <td>2.202570</td>
      <td>180</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6571</th>
      <td>2022-12-29</td>
      <td>108750.475221</td>
      <td>0</td>
      <td>2.503690</td>
      <td>181</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6572</th>
      <td>2022-12-30</td>
      <td>108998.128298</td>
      <td>0</td>
      <td>2.414602</td>
      <td>182</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6573</th>
      <td>2022-12-31</td>
      <td>100746.457400</td>
      <td>0</td>
      <td>2.286391</td>
      <td>183</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>


Finally we can export these dataframes as csv files, which can be found on the github!


```python
# Export combined_daily_interpolation_df to a CSV file
combined_daily_interpolation_df.to_csv('combined_daily_interpolation.csv', index=False)

# Export combined_daily_flagged_df to a CSV file
combined_daily_flagged_df.to_csv('combined_daily_flagged.csv', index=False)
```


```python

```

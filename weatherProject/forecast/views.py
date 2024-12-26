from django.shortcuts import render

# Create your views here.

import requests # This library helps  us to fetch data  from API
import pandas as pd # for handling  and analysing  data
import numpy as np # for  numerical operations
from sklearn.model_selection import train_test_split # to split data  into trainging and testing  data sets
from sklearn.preprocessing import LabelEncoder # to convert categorical data into numerical values
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # models for  classification and regression task
from sklearn.metrics import mean_squared_error # to measure the accurace of our  predictions
from datetime import  timedelta, datetime # to handle  date and time # Import datetime object
import pytz
import os 
from django.contrib.auth.models import User
from django.db import models

API_KEY='16074982138f4a20ef376b34cd18f1f1'
BASE_URL='https://api.openweathermap.org/data/2.5/'#base url for making  API requests



import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_historical_weather(city, start_date, end_date):
    lat, lon = get_lat_lon(city)
    if lat is None or lon is None:
        return {'error': 'Unable to fetch latitude/longitude for city.'}

    # Use daily data instead of hourly
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",  # Daily temperature data
        "temperature_unit": "celsius"
    }

    responses = openmeteo.weather_api(url, params=params)
    return responses[0]

def get_lat_lon(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['coord']['lat'], data['coord']['lon']
    return None, None



def get_currect_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        return {'error': f"API request failed with status {response.status_code}: {response.text}"}

    data = response.json()
    
    # Check if 'name' key exists in the response data
    if 'name' not in data:
        return {'error': "City not found or invalid API response."}
    local_time = datetime.now(pytz.utc) + timedelta(seconds=data['timezone'])
    return {
        'city': data['name'],
        'local_time': local_time.strftime("%H:%M:%S"),
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'pressure': data['main']['pressure'],
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'wind_speed': data['wind']['speed'],
        'clouds': data['clouds']['all'],
        'visibility': data['visibility'],
        
    }

'''
#Fetch Current Weather Data
def get_currect_weather(city):
  url=f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric" #Construct the API request URL
  response= requests.get(url) #send  the get request to API
  data =response.json()
  return {
      'city':data['name'],
      'current_temp':round(data['main']['temp']),
      'feels_like':round(data['main']['feels_like']),
      'temp_min':round(data['main']['temp_min']),
      'temp_max':round(data['main']['temp_max']),
      'pressure':data['main']['pressure'],
      'humidity':round(data['main']['humidity']),
      'description':data['weather'][0]['description'],
      'country':data['sys']['country'],
      'wind_gust_dir':data['wind']['deg'],
      'wind_speed':data['wind']['speed'],
      'clouds':data['clouds']['all'],
      'visibility':data['visibility']


  }
'''

#Read Historical Data
def read_historical_data(filename):
  df=pd.read_csv(filename)
  df=df.dropna()
  df=df.drop_duplicates()
  return df

#Prepare data for training
def prepare_data(data):
  le=LabelEncoder()#create a LabelEncoder instance
  data['WindGustDir']=le.fit_transform(data['WindGustDir'])
  data['RainTomorrow']=le.fit_transform(data['RainTomorrow'])
  #Corrected column names to match those in the DataFrame
  X=data[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp']] # Feature variables
  y=data['RainTomorrow'] # Target variable
  return X,y,le # return feature variable ,target variable and the label encoder

#Train Rain Prediction Model
def train_rain_model(X,y):
  x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
  model=RandomForestClassifier(n_estimators=100,random_state=42)
  model.fit(x_train,y_train)# Train the model
  y_pred=model.predict(x_test)
  print (f"Mean Squared Error for Rain Model")
  print(mean_squared_error(y_test,y_pred))
  return model

#Prepare regression data
def prepare_regression_data(data, feature):
    X, y = [], []  # Initialize lists for features and targets
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    return X, y

  
#Train Regression Model
def train_regression_model(X,y):
  model=RandomForestRegressor(n_estimators=100,random_state=42)
  model.fit(X,y)
  return model

#Predict Future
def predict_future(model,current_value):
  predictions=[current_value]
  for i in range(5):
    next_value=model.predict(np.array([[predictions[-1]]]))
    predictions.append(next_value[0])
  return predictions[1:]

def daily_data(city,date):
        start_date = date  # Set start date here
        end_date = "2024-12-25"    # Set end date here
        historical_data_ = get_historical_weather(city, start_date, end_date)
        
        if isinstance(historical_data_, dict) and 'error' in historical_data_:
            return render(request, 'error.html', {'error': historical_data_['error']})

        # Process daily temperature data
        try:
            daily = historical_data_.Daily()  # Use Daily() method
            print("historical data", daily)
            daily_temperatures = pd.DataFrame({
                'date': pd.to_datetime(daily.Time(), unit="s", utc=True),
                'temperature_max': daily.Variables(0).ValuesAsNumpy(),
                'temperature_min': daily.Variables(1).ValuesAsNumpy(),
                'temperature_mean': daily.Variables(2).ValuesAsNumpy()
            })
            #print("all temps",daily_temperatures)
        except AttributeError:
            return render(request, 'error.html', {'error': 'Daily data is not available.'})

        # Calculate daily averages (already done correctly)
        daily_temperatures['temperature_avg'] = daily_temperatures[['temperature_max', 'temperature_min']].mean(axis=1)

        # Get the day of the week for each date
        daily_temperatures['day_of_week'] = daily_temperatures['date'].dt.strftime('%a')  # Get day of the week

        # Group by 'day_of_week' and calculate the mean of 'temperature_avg' for each day
        daily_avg_temp = daily_temperatures.groupby('day_of_week')['temperature_avg'].mean().reset_index()
        #print("daily avg temp")
        #print(daily_avg_temp)
        # Sort days of the week to display them in order (Mon, Tue, Wed, etc.)
        ordered_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_avg_temp['day_of_week'] = pd.Categorical(daily_avg_temp['day_of_week'], categories=ordered_days, ordered=True)
        daily_avg_temp = daily_avg_temp.sort_values('day_of_week')
        #print("##############daily avg temp")
        #print(daily_avg_temp)
        # Format the temperatures to the desired format
        formatted_daily_temperatures = [
            (round(temp, 1), day)
            for temp, day in zip(daily_avg_temp['temperature_avg'], daily_avg_temp['day_of_week'])
        ]
        print("................TYPE.........................",type(formatted_daily_temperatures))
        return formatted_daily_temperatures
   
#Weather Analysis Function
def weather_view(request):
    if request.method=='POST':
        city=request.POST.get('city')
        current_weather=get_currect_weather(city)

       ##########################################
        # Get historical data for a range of dates
        # Get historical data for a range of dates
        
        date = ["2024-12-21", "2024-12-22", "2024-12-23", "2024-12-24", "2024-12-25"]
        formatted_daily_temperatures = {}

         # Loop through each date and get the corresponding day of the week and average temperature
        for single_date in date:
            result = daily_data(city, single_date)  # daily_data returns a 2D list like [['Mon', 11.1], ...]
            print(result)
            # Assuming each element in result is a tuple with (temperature_avg, day_of_week)
            for temperature_avg, day_of_week in result:  # This will unpack each tuple correctly
                formatted_daily_temperatures[single_date] = {'day_of_week': day_of_week, 'temperature_avg': temperature_avg}

            print(formatted_daily_temperatures)    
# Now formatted_daily_temperatures will have the date as the key and a dictionary with the day and temperature as the value


# Now formatted_daily_temperatures will have the date as key and a dictionary with 'day_of_week' and 'temperature_avg' as values

# Now formatted_daily_temperatures will contain the results from daily_data() for each date


       #########################################
       
        #load historical data
        csv_path= os.path.join('D:\\MS Assignments\\Python_Assignments\\Project\\weather.csv')
        historical_data= read_historical_data(csv_path)

        #prepare and train the rain prediction model
        X,y,le=prepare_data(historical_data)
        rain_model=train_rain_model(X,y)

        wind_deg=current_weather['wind_gust_dir']%360
        compass_points=[
            ('N',0,11.25),('NNE',11.25,33.75),('NE',33.75,56.25),
            ("ENE",56.25,78.75),("E",78.75,101.25),("ESE",101.25,123.75),
            ("SE",123.75,146.25),("SSE",146.25,168.75),("S",168.75,191.25),
            ("SSW",191.25,213.75),("SW",213.75,236.25),("WSW",236.25,258.75),
            ("W",258.75,281.25),("WNW",281.25,303.75),("NW",303.75,326.25),
            ("NNW",326.25,348.75)
        ]
        try:
            compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
        except StopIteration:
            compass_direction = 'Unknown'  # Or some other default value, depending on your requirements

        compass_direction_encoded=le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1
        current_data={
            'MinTemp':current_weather['temp_min'],
            'MaxTemp':current_weather['temp_max'],
            'WindGustDir':compass_direction_encoded,
            'WindGustSpeed':current_weather['wind_speed'],
            'Humidity':current_weather['humidity'],
            'Pressure':current_weather['pressure'],
            'Temp': current_weather['current_temp']
        }

        current_df=pd.DataFrame([current_data])

        #Train prediction
        rain_prediction=rain_model.predict(current_df)

        #Prepare regression model for temperature and humidity
        X_temp,y_temp=prepare_regression_data(historical_data,'Temp')
        X_hum,y_hum=prepare_regression_data(historical_data,'Humidity')
        temp_model=train_regression_model(X_temp,y_temp)
        hum_model=train_regression_model(X_hum,y_hum)

        #Predict future temperature  and humidity
        future_temp=predict_future(temp_model,current_weather['current_temp'])
        future_hum=predict_future(hum_model,current_weather['humidity'])

        #Prepare  time  for future  prediction

        timezone=pytz.timezone('Asia/Karachi')
        now= datetime.now(timezone)
        next_hour=now+timedelta(hours=1)
        next_hour=next_hour.replace(minute=0,second=0,microsecond=0)
        future_times=[(next_hour+timedelta(hours=i)).strftime("%H:00") for i in range(5)]
         
         #store each value seperately
        time1, time2, time3, time4, time5=future_times
        temp1,temp2,temp3,temp4,temp5=future_temp
        hum1, hum2, hum3, hum4, hum5= future_hum
        #pass to template
        context = {
          'location':city,
          'current_temp':current_weather['current_temp'],
          'MinTemp': current_weather['temp_min'],
          'MaxTemp':current_weather['temp_max'],
          'feels_like':current_weather['feels_like'],
          'humidity':current_weather['humidity'] ,   
          'clouds':current_weather['clouds'],
          'description':current_weather['description'],
          'city':current_weather['city'],
          'country':current_weather['country'],
          'time':datetime.now(),
          'date': datetime.now().strftime("%B %d, %Y"),
          'wind':current_weather['wind_speed'],
           'local_time':current_weather['local_time'],
          'pressure':current_weather['pressure'],
          'visibility': current_weather['visibility'],
          'location': city,
          'current_temp': current_weather['current_temp'],
          'description': current_weather['description'],
          'weather_condition': current_weather['description'].lower(),
        
         
          'time1': time1,
          'time2':time2,
          'time3':time3,
          'time4':time4,
          'time5':time5,
          
          'temp1':f"{round(temp1,1)}",
          'temp2':f"{round(temp2,1)}",
          'temp3':f"{round(temp3,1)}",
          'temp4':f"{round(temp4,1)}",
          'temp5':f"{round(temp5,1)}",

           "hum1": f"{round(hum1,1)}",
           "hum2": f"{round(hum2,1)}",
           "hum3": f"{round(hum3,1)}",
           "hum4": f"{round(hum4,1)}",
           "hum5": f"{round(hum5,1)}",
 ################################
           #'daily_temperatures': daily_temperatures.to_html(classes="table table-striped"),
           'daily_temperatures': formatted_daily_temperatures,
            #"start_date" : start_date , # Set start date here
          #" end_date" : end_date  
                            }
        return render(request,'weather.html',context)
    return render(request,'weather.html')




'''
def weather_view(request):
    if request.method=='POST':
        city=request.POST.get('city')
        current_weather=get_currect_weather(city)

       ##########################################
        # Get historical data for a range of dates
        start_date = "2024-12-18"  # Set start date here
        end_date = "2024-12-23"    # Set end date here
        historical_data_ = get_historical_weather(city, start_date, end_date)

        if isinstance(historical_data_, dict) and 'error' in historical_data_:
            return render(request, 'error.html', {'error': historical_data_['error']})

        # Process daily temperature data
        try:
            daily = historical_data_.Daily()  # Use Daily() method
            daily_temperatures = pd.DataFrame({
                'date': pd.to_datetime(daily.Time(), unit="s", utc=True),
                'temperature_max': daily.Variables(0).ValuesAsNumpy(),
                'temperature_min': daily.Variables(1).ValuesAsNumpy(),
                'temperature_mean': daily.Variables(2).ValuesAsNumpy()
            })
            # Use the DataFrame as needed
        except AttributeError:
            return render(request, 'error.html', {'error': 'Daily data is not available.'})

        # Calculate daily averages
        daily_temperatures['temperature_avg'] = daily_temperatures[['temperature_max', 'temperature_min']].mean(axis=1)

       #########################################
       
        #load historical data
        csv_path= os.path.join('D:\\MS Assignments\\Python_Assignments\\Project\\weather.csv')
        historical_data= read_historical_data(csv_path)

        #prepare and train the rain prediction model
        X,y,le=prepare_data(historical_data)
        rain_model=train_rain_model(X,y)

        wind_deg=current_weather['wind_gust_dir']%360
        compass_points=[
            ('N',0,11.25),('NNE',11.25,33.75),('NE',33.75,56.25),
            ("ENE",56.25,78.75),("E",78.75,101.25),("ESE",101.25,123.75),
            ("SE",123.75,146.25),("SSE",146.25,168.75),("S",168.75,191.25),
            ("SSW",191.25,213.75),("SW",213.75,236.25),("WSW",236.25,258.75),
            ("W",258.75,281.25),("WNW",281.25,303.75),("NW",303.75,326.25),
            ("NNW",326.25,348.75)
        ]
        try:
            compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
        except StopIteration:
            compass_direction = 'Unknown'  # Or some other default value, depending on your requirements

        compass_direction_encoded=le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1
        current_data={
            'MinTemp':current_weather['temp_min'],
            'MaxTemp':current_weather['temp_max'],
            'WindGustDir':compass_direction_encoded,
            'WindGustSpeed':current_weather['wind_speed'],
            'Humidity':current_weather['humidity'],
            'Pressure':current_weather['pressure'],
            'Temp': current_weather['current_temp']
        }

        current_df=pd.DataFrame([current_data])

        #Train prediction
        rain_prediction=rain_model.predict(current_df)

        #Prepare regression model for temperature and humidity
        X_temp,y_temp=prepare_regression_data(historical_data,'Temp')
        X_hum,y_hum=prepare_regression_data(historical_data,'Humidity')
        temp_model=train_regression_model(X_temp,y_temp)
        hum_model=train_regression_model(X_hum,y_hum)

        #Predict future temperature  and humidity
        future_temp=predict_future(temp_model,current_weather['current_temp'])
        future_hum=predict_future(hum_model,current_weather['humidity'])

        #Prepare  time  for future  prediction

        timezone=pytz.timezone('Asia/Karachi')
        now= datetime.now(timezone)
        next_hour=now+timedelta(hours=1)
        next_hour=next_hour.replace(minute=0,second=0,microsecond=0)
        future_times=[(next_hour+timedelta(hours=i)).strftime("%H:00") for i in range(5)]
         
         #store each value seperately
        time1, time2, time3, time4, time5=future_times
        temp1,temp2,temp3,temp4,temp5=future_temp
        hum1, hum2, hum3, hum4, hum5= future_hum
        #pass to template
        context = {
            'city': city,
            'current_temp': current_weather['current_temp'],
            'weather_data': {
                'humidity': current_weather['humidity'],
                'pressure': current_weather['pressure'],
                'temp_max': current_weather['temp_max'],
                'temp_min': current_weather['temp_min'],
            },
            'future_temperatures': {
                'time1': time1, 'time2': time2, 'time3': time3, 'time4': time4, 'time5': time5,
                'temp1': temp1, 'temp2': temp2, 'temp3': temp3, 'temp4': temp4, 'temp5': temp5,
            },
            'historical_temperatures': daily_temperatures.to_html(classes="table table-striped"),
            'start_date': start_date,
            'end_date': end_date,
        }
        return render(request,'weather.html',context)
    return render(request,'weather.html')


'''
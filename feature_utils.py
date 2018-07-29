import pandas as pd
from day_of_week import vectorized_dayofweek

#round hour to nearest hour? will have to check about crossing into the next or previous day
def add_hour(df):
    new_test = df.pickup_datetime.str.split().str.get(-2).str.split(':').str.get(0)
    hours = pd.to_numeric(new_test, errors='coerce')
    df['hour'] = hours

def add_day(df):
    new_test = df.pickup_datetime.str.split().str.get(0).str.split('-')
    year = pd.to_numeric(new_test.str.get(0),errors='coerce')
    month = pd.to_numeric(new_test.str.get(1),errors='coerce')
    day = pd.to_numeric(new_test.str.get(2),errors='coerce')
    final_day = vectorized_dayofweek(day.values,month.values,year.values)
    df['day'] = final_day

def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

def add_perimeter_distance(df):
    df['perimeter_distance'] = (df.dropoff_longitude - df.pickup_longitude).abs() + (df.dropoff_latitude - df.pickup_latitude).abs()
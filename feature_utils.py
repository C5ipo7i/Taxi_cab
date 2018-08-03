import pandas as pd
import numpy as np
from day_of_week import vectorized_dayofweek

#round hour to nearest hour? will have to check about crossing into the next or previous day
def add_hour(df):
    #168 categories, 24*7
    data = df.pickup_datetime.str.split().str.get(-2).str.split(':').str
    hour = data.get(0)
    hour_num = pd.to_numeric(hour, errors='coerce')
    minute = data.get(1).str[0]
    minute_num = pd.to_numeric(minute, errors='coerce')
    hours = np.add(np.multiply(hour_num.values,7),minute_num)
    df['hour'] = hours

    #24 categories
    #new_test = df.pickup_datetime.str.split().str.get(-2).str.split(':').str.get(0)
    #hours = pd.to_numeric(new_test, errors='coerce')
    #df['hour'] = hours

def add_day(df):
    new_test = df.pickup_datetime.str.split().str.get(0).str.split('-')
    year = pd.to_numeric(new_test.str.get(0),errors='coerce')
    #print(max(year),'max year')
    #print(min(year),'min year')
    month = pd.to_numeric(new_test.str.get(1),errors='coerce')
    day = pd.to_numeric(new_test.str.get(2),errors='coerce')
    final_day = vectorized_dayofweek(day.values,month.values,year.values)
    df['day'] = final_day

def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

def add_perimeter_distance(df):
    df['perimeter_distance'] = (df.dropoff_longitude - df.pickup_longitude).abs() + (df.dropoff_latitude - df.pickup_latitude).abs()

def add_location_categories(df,decimals):
    #clip decimals at a given point. Can divide by a constant to get a more fluid distribution of boxes
    temp_long = df.dropoff_longitude + 72
    temp_lat = df.dropoff_latitude - 40
    pickup_long = df.pickup_longitude + 72
    pickup_lat = df.pickup_latitude - 40
    check = temp_long.values
    drop_long_key = np.round(temp_long.values,decimals)
    drop_lat_key = np.round(temp_lat.values,decimals)
    pickup_long_key = np.round(pickup_long.values,decimals)
    pickup_lat_key = np.round(pickup_lat.values,decimals)
    #need to times the keys so that they are all ints, take into account the negative long values
    scaled_drop_long = (drop_long_key*-(10**decimals)).astype(np.int64)
    scaled_drop_lat = (drop_lat_key*(10**decimals)).astype(np.int64)
    scaled_pick_long = (pickup_long_key*-(10**decimals)).astype(np.int64)
    scaled_pick_lat = (pickup_lat_key*(10**decimals)).astype(np.int64)
    #take long * max lat+1 value + lat = region num
    #max Y = 2
    #Take X axis * max Y value + 1, + y = value
    max_y = (2*10**decimals)+1
    print(max_y,'maxy')
    dropoff_regions = np.add(np.multiply(scaled_drop_long,max_y),scaled_drop_lat)
    pickup_regions = np.add(np.multiply(scaled_pick_long,max_y),scaled_pick_lat)
    df['pickup_region'] = pickup_regions
    df['dropoff_region'] = dropoff_regions

def add_holidays(df,num_rows):
    #2009 min year, 2015 max year
    from pandas.tseries.holiday import USFederalHolidayCalendar
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2009-01-01',end='2015-12-31').to_pydatetime()
    holidays.reshape(1,holidays.shape[0])
    dates = df.pickup_datetime
    dates.values.reshape(dates.shape[0],1)
    bools = np.array(dates.values[:,None] == holidays)
    holidays = np.sum(bools,axis=-1)
    df['holiday'] = holidays

def K_mean_regions():
    pass
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
    hours = np.add(np.multiply(hour_num.values,6),minute_num)
    df['hour'] = hours

def add_24_hour(df):
    #24 categories
    new_test = df.pickup_datetime.str.split().str.get(-2).str.split(':').str.get(0)
    hours = pd.to_numeric(new_test, errors='coerce')
    df['hour'] = hours

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
    #check = temp_long.values
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

def add_holidays(df):
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
    print('finished with holidays')

def add_K_mean_regions(df,clusters):
    from sklearn.cluster import KMeans
    kmeans_pickup = KMeans(n_clusters=clusters)
    kmeans_dropoff = KMeans(n_clusters=clusters)
    X_dropoff = pd.concat([df.dropoff_longitude,df.dropoff_latitude],axis=1)
    X_pickup = pd.concat([df.pickup_longitude,df.pickup_latitude],axis=1)
    kmeans_pickup = kmeans_pickup.fit(X_pickup.iloc[:1000000])
    pickup_labels = kmeans_pickup.predict(X_pickup)
    kmeans_dropoff = kmeans_dropoff.fit(X_dropoff.iloc[:1000000])
    dropoff_labels = kmeans_dropoff.predict(X_dropoff)
    #C = kmeans.cluster_centers_ #gives location of the cluster centers
    df['pickup_clusters'] = pickup_labels
    df['dropoff_clusters'] = dropoff_labels
    print('finished with Kmean_regions')
    return kmeans_pickup.cluster_centers_,kmeans_dropoff.cluster_centers_

def add_K_mean_grid_routes(df,routes):
    from sklearn.cluster import KMeans
    kmeans_route = KMeans(n_clusters=routes)
    X_route = pd.concat([df.pickup_region,df.dropoff_region],axis=1)
    kmeans_route = kmeans_route.fit(X_route.iloc[:1000000])
    route_labels = kmeans_route.predict(X_route)
    #C = kmeans.cluster_centers_ #gives location of the cluster centers
    df['route_grid_clusters'] = route_labels
    print('finished with Kmean_grid_routes')
    return kmeans_route.cluster_centers_

def add_K_mean_cluster_routes(df,routes):
    from sklearn.cluster import KMeans
    kmeans_route = KMeans(n_clusters=routes)
    X_route = pd.concat([df.pickup_clusters,df.dropoff_clusters],axis=1)
    kmeans_route = kmeans_route.fit(X_route.iloc[:1000000])
    route_labels = kmeans_route.predict(X_route)
    #C = kmeans.cluster_centers_ #gives location of the cluster centers
    df['route_cluster_routes'] = route_labels
    print('finished with Kmean_cluster_routes')
    return kmeans_route.cluster_centers_

def add_K_mean_routes(df,routes):
    from sklearn.cluster import KMeans
    kmeans_route = KMeans(n_clusters=routes)
    X_route = pd.concat([df.pickup_longitude,df.pickup_latitude,df.dropoff_longitude,df.dropoff_latitude],axis=1)
    kmeans_route = kmeans_route.fit(X_route.iloc[:1000000])
    route_labels = kmeans_route.predict(X_route)
    #C = kmeans.cluster_centers_ #gives location of the cluster centers
    df['route_clusters'] = route_labels
    print('finished with Kmean_latlong_route_clusters')
    return kmeans_route.cluster_centers_

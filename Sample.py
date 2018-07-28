import numpy as np
import pandas as pd
import os

train_df = pd.read_csv('/media/shuza/HDD_Toshiba/Taxi_NYC/train.csv')
train_df.dtypes

def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(train_df)

print(train_df.isnull().sum())

print('Old size: %d' % len(train_df))
train_df = train_df.dropna(how='any', axis = 'rows')
print('New size: %d' % len(train_df))


print('Old size: %d' % len(train_df))
train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]
print('New size: %d' % len(train_df))

def get_input_matrix(df):
    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df))))

train_X = get_input_matrix(train_df)
train_y = np.array(train_df['fare_amount'])

print(train_X.shape)
print(train_y.shape)

(w, _, _, _) = np.linalg.lstsq(train_X,train_y,rcond=None)
print(w)
w_OLS = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)), train_X.T), train_y)
print(w_OLS)

test_df = pd.read_csv('/media/shuza/HDD_Toshiba/Taxi_NYC/test.csv')
print(test_df.dtypes)

add_travel_vector_features(test_df)
test_X = get_input_matrix(test_df)

test_y_predictions = np.matmul(test_X,w).round(decimals = 2)

submission = pd.DataFrame(
    {'key': test_df.key,'fare_amount': test_y_predictions},
    columns = ['key','fare_amount'])

submission.to_csv('submission.csv',index = False)

print(os.listdir('.'))
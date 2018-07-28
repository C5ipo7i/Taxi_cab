import numpy as np
import pandas as pd
import os

"""
Outline:
Weekend/weekday marker
Holiday marker
Split time into 24 classes
Location could also be split into classes

Clean data - remove outliers
"""


def add_hour(df):
    df['hour'] = (int(df.pickup_datetime.to_string().split()[2][:2]))

train_df = pd.read_csv('/media/shuza/HDD_Toshiba/Taxi_NYC/train.csv',nrows=10)
add_hour(train_df)
print(train_df.dtypes)
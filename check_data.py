import pandas as pd 

num_rows = 10000000

with open('/media/shuza/HDD_Toshiba/Taxi_NYC/clean_train.csv') as clean_train_csv:
    df = pd.read_csv(clean_train_csv)

print(list(df))
print(df.head(10))
print(df.tail(10))
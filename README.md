# Taxi_cab
Kaggle taxi cab NYC competition (for knowledge!)

My first Kaggle competition. Wanted to learn about big data sets and how certain learning techniques stacked up against Kagglers in general.

# The Data
Lat/Long data for passenger pickup and dropoff
Number of passengers picked up
Fare charged

# Cleaning the data
clip outliers,
convert date to day of the week
convert time to 24 classes. (chunk by hour)
add L2 distance

I also added a holiday check but there were none.

# The models
Tested a number of different models. Some with embeddings and using a grid to chunk locations. A simplier NN scored higher.

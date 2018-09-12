import sys
import os

training_dir_alex = os.path.join(os.path.dirname(sys.argv[0]), "Taxi_NYC")
clean_data_path_alex = os.path.join(training_dir_alex, "LSTM_dual_head_V2")

print(clean_data_path_alex)
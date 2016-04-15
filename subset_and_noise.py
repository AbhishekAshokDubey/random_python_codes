# code to add noise to a subset (columns) of data

import pandas as pd
import numpy as np
import random
import math

sample_column_count = 8;
max_percentage_noise_around_mean = 10;

noise_bucket_levels = 4; # precision of float value = noise_bucket_levels+2

datafile_path = r"C:\Users\*";
df = pd.read_csv(datafile_path, skipinitialspace=True, parse_dates=[0], infer_datetime_format=True);
total_column_count = df.shape[1]

slected_col_indx = random.sample(range(1, total_column_count), sample_column_count)
slected_col_names = df.columns[slected_col_indx]
df = df[slected_col_names]

sample_limit_value = max_percentage_noise_around_mean * int(math.pow(10,noise_bucket_levels));
for i,col in enumerate(df.columns):
    random_percentage = np.random.choice(range(-sample_limit_value, sample_limit_value), df.shape[0])/(100*math.pow(10,noise_bucket_levels));
    df[col] = df[col] + df[col].mean()*random_percentage;

df.to_csv(r"C:\Users\*")

#import numpy as np
#noise = np.random.normal(0,1,100)
## 0 is the mean of the normal distribution you are choosing from
## 1 is the standard deviation of the normal distribution
## 100 is the number of elements you get in array noise

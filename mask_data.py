# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 00:31:39 2017

@author: ADubey4
"""

import pandas as pd
import numpy as np
import numbers
from random import choice
from string import ascii_uppercase
import pickle
from math import ceil

import os

os.chdir(r"path");

not_to_touch_column = ["field_ticket", "start_date", "end_date", "job_days"]

file_name = "wl_eq_rp_train_ftl"

noise_type = 'normal' # 'uniform', 'normal', 'laplace'
noise_var_1 = 0 # mean or min
noise_var_2 = None # std or max, make None for scaled values
noise_percentage_of_std = 1; # if noise_var_2 is None
data_file_path = r"path\\" + file_name + ".csv";
outlier_percentage = 1;
outlier_std_range = [3,5]; # noise_var_2 for the outlier distribution

################# All functions #################
###############################################################################
# Def: Random noise data of goven 'size'
def gen_rand(noise_type="normal",val_1 = 0, val_2 = 1, size = 1):
    if (np.isnan(val_1) or np.isnan(val_2)):
        val_1 = 0;
        val_2 = 1;
    if(val_1>=0 and val_2==0): # constant std case
        val_2 = 0.001;
    if noise_type == "uniform":
        noise = np.random.uniform(low=val_1, high=val_2, size=size)
    elif noise_type == "normal":
        noise = np.random.normal(loc=val_1, scale=val_2, size=size)
    elif noise_type == "laplace":
        noise = np.random.laplace(loc=val_1, scale=val_2, size=size)
    return noise;

# Def: function to generate the dict of masking strings
# TODO: adding assert for infinite loop, in case of possibly continous variable
def gen_replacement_dict(df_passed):
    df = df_passed.copy();
    masking_dict = {}
    masking_dict[np.nan] = ""
    value_set = set()
    for col in df:
        unique_val = pd.unique(df[col])
        for val in unique_val:
            if val not in masking_dict:
                while True:
                    replace = ''.join(choice(ascii_uppercase) for _ in range(12))
                    if replace not in value_set:
                        masking_dict[val] = replace;
                        value_set.add(replace);
                        break;
    return masking_dict;

# Def function to replace all the values by masking dictionary
def replace_with_dict(df_passed, masking_dict):
    df = df_passed.copy();
    return df.applymap(lambda x: masking_dict[x]);

# Def: function to add dummy column as linear combination of other columns
# TODO: add multiple columns with one call
def make_dependent_column(df_passed, new_col_name= "col"):
    df = df_passed.copy();
    column_names = df.columns;
    if len(column_names) == 0:
        return None, None
    elif len(column_names) == 1:
        df_return = np.random.randint(1,10) * df.ix[:,0] +  np.random.randint(1,ceil(np.mean(df.ix[:,0])))
        return df_return, ""
    else:
        no_of_columns = np.random.randint(1,len(column_names),1)
        random_indexes = np.random.choice(range(len(column_names)),no_of_columns, replace=False)
        df_consider = df.ix[:,random_indexes]
        for i,col_name in enumerate(df_consider):
            if i==0:continue;
            df_consider.ix[:,0] = df_consider.ix[:,0] + df_consider[col_name]
        df_return = df_consider.ix[:,0]
        df_return.name = new_col_name
        return df_return, df_consider.columns

# TODO: chnage from pass by reference to pass by variable
#       Also add values on the negative sides
def add_outliers(df_passed, noise_type="uniform"):
    df = df_passed.copy();
    row_count = len(df)
    outlier_count = ceil((outlier_percentage * row_count)/100.0)
    for col_name in df:
        random_indexes = np.random.choice(range(row_count),outlier_count, replace=False);
        mean_temp = np.mean(df[col_name]);
        std_temp = np.std(df[col_name]);
        outliers = gen_rand(noise_type,
                            mean_temp + (outlier_std_range[0]*std_temp),
                            mean_temp + (outlier_std_range[1]*std_temp),
                            size = outlier_count)
        df.ix[random_indexes,col_name] = outliers;
    return df

#df =  pd.read_csv(data_file_path)
#df_num = df._get_numeric_data().copy()
#add_outliers(df_num)
###############################################################################


################# user code #################
###############################################################################
df =  pd.read_csv(data_file_path,encoding="ISO-8859-1")
col_names = df.columns.drop(not_to_touch_column)
data_points_count = len(df)

num_cols = df._get_numeric_data().columns
char_cols = df.columns.drop(num_cols)

to_drop_numeric_columns = [x for x in num_cols if x in not_to_touch_column];
num_cols = num_cols.drop(to_drop_numeric_columns)

to_drop_char_columns = [x for x in char_cols if x in not_to_touch_column];
char_cols = char_cols.drop(to_drop_char_columns)

df[num_cols] = add_outliers(df[num_cols])

for col_name in num_cols:
    if isinstance(noise_var_2, numbers.Number):
        noise_var_2_temp = noise_var_2
    else:
        noise_var_2_temp = np.std(df[col_name]) * (noise_percentage_of_std/100)
    df[col_name] = df[col_name] + gen_rand(val_1 = noise_var_1, val_2 = noise_var_2_temp, size=data_points_count)

df_char = df[char_cols]
masking_dict = gen_replacement_dict(df_char)
df_char = replace_with_dict(df_char,masking_dict)

for col_name in char_cols:
    df[col_name] = df_char[col_name];

df_dep, merged_columns = make_dependent_column(df[num_cols], "new_col")
if df_dep is not None:
    df = pd.concat([df,df_dep],axis=1)

pickle.dump( masking_dict, open( r"path\\" + file_name +"_masking_dict.p", "wb" ) )
# masking_dict = pickle.load( open( "masking_dict.p", "rb" ) )
df.to_csv(r"path\\"+ file_name +"_masked.csv",index=False);

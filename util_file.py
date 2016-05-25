# -*- coding: utf-8 -*-
"""
Created: 5/04/2016
@author: ADubey4
"""

import numpy as np
import pandas as pd
import math

# function to get the first and third quatile
# Set extact_quartile_value = true, if we need the quartile value to be one of the data points 
def get_Q1_and_Q3(df, extact_quartile_value):
    if not extact_quartile_value:
        Q1 = np.percentile(df,25,axis=0);
        Q3 = np.percentile(df,75,axis=0);
    else:
        df_sorted = pd.DataFrame();
        for col_name in df.columns:
            df_sorted[col_name] = sorted(df[col_name]);
        row_count = df.shape[0];
        
        Q1_index = math.floor((row_count)*0.25);
        Q1 = df_sorted.iloc[Q1_index]
        
        #to reduce the Q3 index, can be seen only for row_count%4 == 0 case.
        #This can also be removed
        row_count = (row_count - 1) if row_count % 4 == 0 else row_count
        
        Q3_index = math.floor((row_count)*0.75); 
        Q3 = df_sorted.iloc[Q3_index]
        
    return Q1,Q3;

# Normalize the data
def normalize_data(df, use_std = True):
    if use_std:
        return (df - df.mean()) / (df.std());
    else:
        return (df - df.mean()) / (df.max() - df.min()); #Prob: constant val column

# Once the outliers are detected, remove the outliers with respected mean values
def clean_mask_data(df, outlier_mask, do_norm, negative_norm, replace_with_mean):
    if replace_with_mean:
        non_outlier_values = ~outlier_mask * df;
        mean = non_outlier_values.sum() / (non_outlier_values != 0).sum();
        #    mean = non_outlier_values.sum() / ((non_outlier_values != 0).sum()).replace(0,1);
        df_outlier_mean = outlier_mask * mean;
        df_correct_val = ~outlier_mask * df;
        df_clean = df_outlier_mean + df_correct_val;

    else:
        df_clean = df;
        col_count = 1;
        if len(outlier_mask.shape) == 1:
            outlier_indexs = np.where(outlier_mask == True)
            for i in outlier_indexs[0]:
                if i == 0:
                    df_clean.iloc[i] = df_clean.iloc[i+1]
                elif i == len(outlier_indexs[0])-1:
                    df_clean.iloc[i] = df_clean.iloc[i-1]
                else:
                    df_clean.iloc[i] = (df_clean.iloc[i-1] + df_clean.iloc[i+1])/2
        elif len(outlier_mask.shape) == 2:
            col_count = outlier_mask.shape[1];
            for col_num in range(col_count):
                outlier_indexs = np.where(outlier_mask.iloc[:,col_num] == True)
                for i in outlier_indexs[0]:
                    if i == 0:
                        df_clean.iloc[i][col_num] = df_clean.iloc[i+1][col_num]
                    elif i == len(outlier_indexs[0])-1:
                        df_clean.iloc[i][col_num] = df_clean.iloc[i-1][col_num]
                    else:
                        df_clean.iloc[i][col_num] = (df_clean.iloc[i-1][col_num] + df_clean.iloc[i+1][col_num])/2

    if do_norm:
        return normalize_data(df_clean,negative_norm)
    else:
        return df_clean            


def func_outlier(df, method = "STD", method_parm = 1.5, only_outlier = True, do_norm = False, negative_norm = True, replace_with_mean = True, extact_quartile_value = False):
    if method == "STD":
        outlier_mask = abs(df - df.mean()) > (method_parm * df.std());
    elif method == "IQR":
        Q1, Q3 = get_Q1_and_Q3(df, extact_quartile_value)
        IQR = Q3 - Q1;
        outlier_mask = (df > (Q3 + method_parm * IQR)) | (df < (Q1 - method_parm * IQR));
    if only_outlier:
        return pd.DataFrame, outlier_mask;
    else:
        df_new = clean_mask_data(df, outlier_mask, do_norm, negative_norm, replace_with_mean);
        return df_new, outlier_mask;

##########################
#Example to show the utiltiy of the package
# a = pd.DataFrame([[1,2,3],[4,5,6],[10,8,90],[4,20,2]])
# b1, b2 = replace_sd_based_outlier_with_mean(a,1);
# c1, c2 = replace_IQR_based_outlier_with_mean(a,1.5);

################################################################################
##Way tp import the utiltiy (temporarily) in your code
#def import_file(full_path_to_module):
#    try:
#        import os
#        module_dir, module_file = os.path.split(full_path_to_module)
#        module_name, module_ext = os.path.splitext(module_file)
#        save_cwd = os.getcwd()
#        os.chdir(module_dir)
#        module_obj = __import__(module_name)
#        module_obj.__file__ = full_path_to_module
#        globals()[module_name] = module_obj
#        os.chdir(save_cwd)
#    except:
#        raise ImportError
#
#import_file(r"C:\Users\Adubey4\Desktop\AD_Python_utility_funcs\util_abhi.py")

#def num_abhi(n):
#    print(math.floor(n*0.25))
#    n = (n - 1) if n % 4 == 0 else n
#    print(math.floor(n*0.75))

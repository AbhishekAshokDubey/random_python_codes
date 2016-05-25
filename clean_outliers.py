#!/usr/bin/python

import numpy as np
import pandas as pd
import math

from util_file import func_outlier

datafile_path = './pradeep_sensor_data_file.csv'
col_name_to_normalise_list = ["Temp", "DewPt", "CldCvr", "WndSpd", "Precip", "Inflow"];
col_name_to_remove_outlier_list = ["Temp", "DewPt", "CldCvr", "WndSpd", "Precip", "Inflow"];
carryOn_col = ["Index"];

outliers_preprocess_option = {}
outliers_preprocess_option["outliers"] = True
########
outliers_preprocess_option["outliers_removal_mtd"] = "IQR" # "IQR" or "STD"
outliers_preprocess_option["IQR_STD_parrm"] = 1.5
#outliers_preprocess_option["extact_quartile_value"] = False
########
outliers_preprocess_option["only_outlier"] = False
########
outliers_preprocess_option["replace_outlier_with_mean"] = False # False: interpolation & True: mean
########
outliers_preprocess_option["norm_data"] = True
outliers_preprocess_option["negative_norm_values"] = True
########

df = pd.read_csv(datafile_path, skipinitialspace=True);

if outliers_preprocess_option["outliers"]:
    df, outlier_mask = func_outlier(df[col_name_to_remove_outlier_list],\
    method = outliers_preprocess_option["outliers_removal_mtd"],\
    method_parm = outliers_preprocess_option["IQR_STD_parrm"],\
    only_outlier = outliers_preprocess_option["only_outlier"], 
    do_norm = outliers_preprocess_option["norm_data"], 
    negative_norm = outliers_preprocess_option["negative_norm_values"], 
    replace_with_mean = outliers_preprocess_option["replace_outlier_with_mean"], 
    extact_quartile_value = False)

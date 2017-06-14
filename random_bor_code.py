# -*- coding: utf-8 -*-
"""
Created on Tue May  3 21:18:26 2016

About the file: 
                This file can be used for both
                WL assets as well as WL people predictions
                
                This code makes the dictionary to get all the asset type/
                people type and then uses ML to predict the individuals

@author: ADubey4
"""
## Code for the changing the current directory to source directory
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#import sys
#sys.modules[__name__].__dict__.clear()

## All the libraries required
##########################################################################
import itertools
import pandas as pd
import numpy as np
import math
import pydotplus
# pip install pydotplus
from sklearn import tree
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.externals.six import StringIO
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import grid_search

#supressing silly pandas warning###
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
pd.options.mode.chained_assignment = None 
###################################


################### Set attributes #########################

#datafile_path = r"C:\Users\Adubey4\Desktop\Desktop\BOR_DS\ML\WL_Assets_Data_Dump.csv";
#datafile_path = r"C:\Users\Adubey4\Desktop\Desktop\BOR_DS\ML\Wireline_People_Data.csv";
datafile_path = r"WL_Assets_Data_Dump.csv";

use_col = ["SO_Number", "Customer", "Well_Environment", "Well_Type", "AMR_Tool", "HoleSizeRange", "Total_Tool_Days", "Total_Holesection_Duration"];
#use_col = ["SO_Number", "Customer_Name", "Well_Environment", "Well_Type", "Employee_Type", "HOLE_SIZE_RANGE", "Role_Days", "Total Duration"];

attribute_names = ["Well_Environment","Customer","Well_Type","HoleSizeRange"]
#attribute_names = ["Well_Environment","Customer_Name","Well_Type","HOLE_SIZE_RANGE"]
most_granular_feature = "AMR_Tool"
#most_granular_feature = "Employee_Type"

target_var = "Total_Tool_Days";
#target_var = "Role_Days";

activity_var = "Total_Holesection_Duration";
#activity_var = "Total Duration";

projectNo_col_name = "SO_Number"
train_test_split_ratio = 0.8;


###################### Code begins ##################################

# FOR IQR, outlier removal
def get_Q1_and_Q3(df, extact_quartile_value):
    if extact_quartile_value:
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

def get_outliers(df, IQR_cutOff = 1.5):
    Q1, Q3 = get_Q1_and_Q3(df, True)
    IQR = Q3 - Q1;
    outlier_mask = (df > (Q3 + IQR_cutOff * IQR)) | (df < (Q1 - IQR_cutOff * IQR));
    return outlier_mask;

df = pd.read_csv(datafile_path, skipinitialspace=True, usecols = use_col);
temp = df[target_var] / df[activity_var]
#outlier_mask = get_outliers(df[target_var].copy());
outlier_mask = get_outliers(temp);
print("Removing "+str(sum(outlier_mask))+ " out of "+ str(df.shape[0]))
df = df[~outlier_mask];



df_train, df_test = train_test_split(df, train_size = train_test_split_ratio);
df_train.reset_index(drop=True,inplace=True)

df_test.sort([projectNo_col_name],inplace=True)
df_test.reset_index(drop=True,inplace=True)

# Must skip for round 1 reading
# Dictionary to convert Char variables to columns of data frame
df_factor_data = df[attribute_names + [most_granular_feature]]
df_factor_dat_as_dict = [dict(r.iteritems()) for _, r in df_factor_data.iterrows()]
vectorizer = DictVectorizer()
df_vectorizer = vectorizer.fit(df_factor_dat_as_dict)


# Must skip for round 1 reading
# Convert string variable into columns
def make_char_as_feature(df):
    df_factor_data = df[attribute_names + [most_granular_feature]]
    df_factor_dat_as_dict = [dict(r.iteritems()) for _, r in df_factor_data.iterrows()]
#    vectorizer = DictVectorizer()
#    df_factor_data = vectorizer.fit_transform(df_factor_dat_as_dict)
    df_factor_data = df_vectorizer.transform(df_factor_dat_as_dict)
    #vectorizer.get_feature_names()
    return df_factor_data,vectorizer.get_feature_names()


# Subset data based on a scenario and call "make_char_as_feature" to Convert string variable into columns
def make_scenario_dataset(df, combination_tuple, only_data_subset=False):
    null_attributes = set(attribute_names).difference(set(combination_tuple))
    for null_attr in null_attributes:
        df = df.loc[df[null_attr] == "-"]
    for attr in  list(combination_tuple):
        df = df.loc[df[attr] != "-"]
    df.reset_index(drop=True,inplace=True)
    if not only_data_subset:
        df_factor_data, factor_col_names = make_char_as_feature(df);
        df_X1 = pd.DataFrame(data = df_factor_data.toarray(), columns = factor_col_names)
        df_X2 = pd.concat([df_X1, df[activity_var]], axis = 1)
        #    X.iloc[1][X.iloc[1] == 1]
        df_y = df[target_var]
        return df_X2, df_y, df
    else:
        return df

# Make dictionary for "AC ===> tool_type" combination from train data
def mostGranularAttr_dict_per_ac(df,ac):
    ac_tool_dict = {};
    print("building dict for Scenario.. " + str(ac))
    for index, row in df.iterrows():
        if "#".join(row[ac]) in ac_tool_dict:
            ac_tool_dict["#".join(row[ac])] = ac_tool_dict["#".join(row[ac])] + [row[most_granular_feature]]
        else:
            ac_tool_dict["#".join(row[ac])] = [row[most_granular_feature]] 
    print("done!")
    return ac_tool_dict;
    

# get error and predicted data frame for test data, using trained clf
def get_prediction_error(clf, df, ac_tool_dict, ac):
    i = 0;
    df_predicted = pd.DataFrame(columns = df.columns)
    df_predicted["Predicted"] = 0
    error = 0;
    count = 0;
    while i < df.shape[0]:
        if "#".join(df.iloc[i][ac]) not in ac_tool_dict:
            df_project = df[df[projectNo_col_name] == df.iloc[i][projectNo_col_name]]
            i = i + df_project.shape[0];
            continue;

        all_tools_past = ac_tool_dict["#".join(df.iloc[i][ac])];
        df_project = df[df[projectNo_col_name] == df.iloc[i][projectNo_col_name]]

        tools_in_current = df_project[most_granular_feature]
        tools_to_add_from_past = list(set(all_tools_past).difference(set(tools_in_current)))
        tools_new_in_current = list(set(tools_in_current).difference(set(all_tools_past)))
        tools_in_both = set(tools_in_current).difference(tools_new_in_current)
        
        df_inPast_inCurrent = df_project[df_project[most_granular_feature].isin(tools_in_both)];
        df_notPast_inCurrent = df_project[df_project[most_granular_feature].isin(tools_new_in_current)];
        
        if not len(tools_to_add_from_past):
            df_inPast_notCurrent = pd.DataFrame(columns = df_project.columns);
        else:
            df_inPast_notCurrent = pd.concat([pd.DataFrame(df_project.iloc[0:1])]*len(tools_to_add_from_past)).reset_index(drop=True);
            [df_inPast_notCurrent.set_value(i,most_granular_feature,tool).set_value(i,target_var,0) for i,tool in enumerate(tools_to_add_from_past)]
        
        if df_inPast_inCurrent.shape[0]:
#            print("in in------------"+str(df_inPast_inCurrent.shape[0]))
            X_inPast_inCurrent,_,_ = make_scenario_dataset(df_inPast_inCurrent.copy(), combination_tuple)
            y_predicted_inPast_inCurrent = clf.predict(X_inPast_inCurrent)
            df_inPast_inCurrent["Predicted"] = y_predicted_inPast_inCurrent;
            df_predicted = df_predicted.append(df_inPast_inCurrent);
            count += df_inPast_inCurrent.shape[0];
            temp = df_inPast_inCurrent[target_var].copy()
            temp.loc[temp == 0] = 1 
            error += sum(abs(df_inPast_inCurrent["Predicted"] - df_inPast_inCurrent[target_var]) / temp)

        if df_inPast_notCurrent.shape[0]:
#            print("in not------------"+str(df_inPast_notCurrent.shape[0]))
            X_inPast_notCurrent,_,_ = make_scenario_dataset(df_inPast_notCurrent.copy(), combination_tuple)        
            y_predicted_inPast_notCurrent = clf.predict(X_inPast_notCurrent)
            df_inPast_notCurrent["Predicted"] =  y_predicted_inPast_notCurrent;
            df_predicted = df_predicted.append(df_inPast_notCurrent);

        if df_notPast_inCurrent.shape[0]:
            df_notPast_inCurrent["Predicted"] = 0;
            df_predicted = df_predicted.append(df_notPast_inCurrent);
        
        df_predicted.reset_index(drop=True)        
        i = i + df_project.shape[0];
#        print(i)
    error = error/count;
    return error, df_predicted
        
all_attribute_combination_list = [];
for L in range(1, len(attribute_names)+1):
  for subset in itertools.combinations(attribute_names, L):
    all_attribute_combination_list.append(subset)

ac_tool_dict = {};

# If you want to check output for just one combination, change if to "index == 10 or False"
for index, combination_tuple in enumerate(all_attribute_combination_list):
    if index == 14 or False:
        X_train, y_train, df_train_ac = make_scenario_dataset(df_train.copy(), combination_tuple)
        df_test_ac = make_scenario_dataset(df_test.copy(), combination_tuple, only_data_subset=True)
    
        #clf = tree.DecisionTreeRegressor()
    

#        clf = ensemble.RandomForestRegressor()
#        clf = ensemble.RandomForestRegressor(n_estimators=30,oob_score=True)
#        param_grid = [{'n_estimators' : [5,10,20,30],
#        'bootstrap' : [False]},
#        {'n_estimators' : [5,10,20,30],
#        'oob_score' : [True],
#        'bootstrap' : [True]}]
#        clf_grid = grid_search.GridSearchCV(clf, param_grid, verbose = 10)



#        clf = GradientBoostingRegressor()
        clf = GradientBoostingRegressor(alpha = 0.9, loss='huber', learning_rate=0.3, n_estimators= 300)
#        param_grid = [{'loss' : ['ls','lad','huber', 'quantile'],
#        'learning_rate': [0.1,0.3,0.5],
#        'n_estimators': [50,100,200],
#        'alpha': [0.3,0.6,0.9]}]
#        clf_grid = grid_search.GridSearchCV(clf, param_grid, verbose = 10)


#        clf_grid.fit(X_train, y_train)
#        clf_grid.grid_scores_
#        clf = clf_grid.best_estimator_
#        imp_sorted_features1 = [x for x in sorted(clf_grid.grid_scores_, key=lambda tupple: tupple[1],reverse=True)]


#        dot_data = StringIO() 
#        tree.export_graphviz(clf, out_file=dot_data, feature_names=X_train.columns) 
#        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#        graph.write_pdf(",".join(combination_tuple)+".pdf")
#        print("1")

        clf = clf.fit(X_train, y_train)

        ac_tool_dict = mostGranularAttr_dict_per_ac(df_train_ac, list(combination_tuple));
        error, predicted_frame = get_prediction_error(clf, df_test_ac, ac_tool_dict, list(combination_tuple));
        print("-----"+str(combination_tuple)+"-----")
        print("Error: %.2f %%" % error);
        print("-===========================================-")
#    list(mapper["Well_Environment"].inverse_transform([0]))
#imp_sorted_features1 = [(y,x) for (y,x) in sorted(zip(clf.feature_importances_,X_train.columns), key=lambda pair: pair[0],reverse=True)]
#imp_sorted_features2 = sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), X_train.columns),reverse=True)
        

#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

# # Benchmark Evaluation for Predictive Monitoring of Remaining Cycle Time of Business Processes

from bucketing import get_encoder, get_bucketer
from xgboost.core import XGBoostError
from dataset_manager import DatasetManager
from classifiers import get_classifier, generate_local_explanations
from misc import Debug

from env_vars import MY_WORKSPACE_DIR

# add my working directory to the colab path
import sys

sys.path.append(MY_WORKSPACE_DIR)

import warnings

warnings.filterwarnings('ignore')

from time import sleep

# sklearn library
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

# LIME - Explainability
import lime
import lime.lime_tabular
from lime import submodular_pick;  # not using this but useful later.

# serialise models
from numpy import loadtxt
import pickle

# visualization
# from misc.misc import *
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:90% !important; }</style>"))

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import pylab as pl
from pylab import savefig

import seaborn as sns

sns.set()

from time import time

import itertools
import pickle
import os
import pandas as pd
import numpy as np
import csv
from numpy import array

# ## Predictive Process Monitoring: Remaining Time Prediction
# 
# Given an event log of complete cases of a business process, and a prefix case of the process as obtained from an event stream, we want to predict a performance measure of such prefix case in the future. For example, we may wsnt to predict the time of this case until completion (or remaining time) or its outcomeat completion. A **prediction point** is a point in the future where the performance measure has a predicted value. A prediction is thus based o nthe predictor's knowledge of the history of the process until the prediction point as well as knowledge of the future until the predicted point. The former is warrented by the predictor's **memory** while the latter is based on  the predictor's **forecast**, i.e., predicting the future based based on the trend nd seasonal patern analysis. Finally, the prediction is performed based on  a**prediction** algorithm.
# Since in real-life business processes the amount of uncertainty increases over time, the prediction task becomes more difficult and genersally less acurate. As such, predictions are made up to a specific point of time in the future, i.e., the time zone **h**. The choice of **h** depends on how fast the process evolves and on the prediction goal.

# ### Research Questions
# 
# Research questions analysed in paper
# :
#   - What methods exist for predictive monitoring of remaining time of business processes?
#   - How to classify methods for predictive monitoring of remaining time?
#   - What tyoe of data has been used to evaluate these methods, and from which application domains?
#   - What is the relative performance of these methods?

# ### Methodology
# 
# <img href="https://www.dropbox.com/s/4uj3yll961chfau/predictive_process_monitoring_workflow.png" />
# 

# #### Prefix Bucketing
# 
# Two possible approaches used in machine-learning-based predictive process monitoring:
# 
#   1. train a single predictor on the whole event log;
#   2. employ a multiple predictor apporach by dividing the prefix traces in the historical log into several **buckets** and fitting a separate predictor for each bucket. 
# 
# The four most used bucketing methods in the literature are:
#   1. Zero Bucketing
#   2. Prefix length bucketing
#   3. Cluster bucketing
#   4. State Bucketing
# 
#   
# 
# 

# In[45]:


# #### Prefix Encoding

# In[51]:


# ##### Static Encoder

# In[52]:

# #### Train Classifiers

# ##### Train Classifier XgBoost

# In[70]:


# In[71]:


import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# -----------------------------------------------------------------------
# TODO: in BPIC2011 use only the "other" treatment code and check results
# --------------------------------------------------------------

# SIngle bucket, aggregation encoding
# bucket_method = single
# cls_encoding = agg

# prefix bucket, aggregation encoding
# bucket_method = prefix
# cls_encoding = agg

# prefix bucket, index encoding
# bucket_method = prefix
# cls_encoding = index

# Run LSTM
dataset_ref = "bpic2012w"

bucket_method = "prefix"
bucket_encoding = "agg"

cls_encoding = "agg"
cls_method = "xgb"

results_dir = MY_WORKSPACE_DIR / "results/"

if bucket_method == "state":
    bucket_encoding = "last"

method_name = "%s_%s" % (bucket_method, cls_encoding)

home_dir = MY_WORKSPACE_DIR

if not os.path.exists(os.path.join(home_dir, results_dir)):
    os.makedirs(os.path.join(home_dir, results_dir))

dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011"],
    "bpic2012a": ["bpic2012a"],
    "bpic2012o": ["bpic2012o"],
    "bpic2012w": ["bpic2012w"],
    "bpic2015": ["bpic2015_5"],
    "insurance": ["insurance_activity", "insurance_followup"],
    "bpic2017": ["bpic2017"]
}

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]

# bucketing params to optimize 
if bucket_method == "cluster":
    bucketer_params = {'n_clusters': [2, 4, 6]}
else:
    bucketer_params = {'n_clusters': [1]}

# classification params to optimize
if cls_method == "rf":
    cls_params = {'n_estimators': [250, 500],
                  'max_features': ["sqrt", 0.1, 0.5, 0.75]}

elif cls_method == "xgb":
    cls_params = {'n_estimators': [500],
                  'learning_rate': [0.06],
                  'subsample': [0.8],
                  'max_depth': [3, 5, 7],
                  'colsample_bytree': [0.6, 0.9]}

bucketer_params_names = list(bucketer_params.keys())
cls_params_names = list(cls_params.keys())

outfile = os.path.join(home_dir, results_dir, "cv_results_%s_%s_%s.csv" % (cls_method, method_name, dataset_ref))
print(outfile)

train_ratio = 0.8
random_state = 22
fillna = True
n_min_cases_in_bucket = 30

from preprocessing import add_remtime_column, get_open_cases

from env_vars import MY_WORKSPACE_DIR


def train(*, save_data_chunks: bool = False):
    with open(outfile, 'w') as fout:
        fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (
            "part", "dataset", "method", "cls", ";".join(bucketer_params_names), ";".join(cls_params_names),
            "nr_events", "metric", "score"))

        gen_counter = 0
        nr_events = 0
        exp_indx = 0
        for dataset_name in datasets:

            dataset_manager = DatasetManager(dataset_name)

            # read the data
            data = dataset_manager.read_dataset()
            data = data.groupby(dataset_manager.case_id_col).apply(add_remtime_column,
                                                                   dataset_manager.timestamp_col).reset_index(drop=True)
            data["open_cases"] = get_open_cases(data, dataset_manager.case_id_col, dataset_manager.timestamp_col)

            # split data into train and test
            train_split, _ = dataset_manager.split_data(data, train_ratio)

            # consider prefix lengths until 90% of positive cases have finished
            min_prefix_length = 1
            max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))

            part = 0
            for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train_split, n_splits=3):
                part += 1
                print("Starting chunk %s..." % part)
                sys.stdout.flush()

                # create prefix logs
                dt_train_prefixes = dataset_manager.generate_prefix_data(train_chunk, min_prefix_length,
                                                                         max_prefix_length)
                dt_test_prefixes = dataset_manager.generate_prefix_data(test_chunk, min_prefix_length,
                                                                        max_prefix_length)

                print(dt_train_prefixes.shape)
                print(dt_test_prefixes.shape)

                # creating a dictionary to store explanations ####ADDED BY RENUKA
                exp_dict = dict()

                # #####################################################################
                # GET DATASET BY CHUNKS

                if save_data_chunks:
                    df_train = pd.DataFrame(dt_train_prefixes)
                    df_train.to_csv(
                        str(MY_WORKSPACE_DIR) + "/XGBoost/buckets_" + bucket_method + "_" + cls_encoding + "/train/chunk" + str(
                            part) + "/train_" + dataset_ref + "_p" + str(part) + ".csv", index=False)

                    df_test = pd.DataFrame(dt_test_prefixes)
                    df_test.to_csv(
                        str(MY_WORKSPACE_DIR) + "/XGBoost/buckets_" + bucket_method + "_" + cls_encoding + "/test/chunk" + str(
                            part) + "/test_" + dataset_ref + "_p" + str(part) + ".csv", index=False)

                # #####################################################################

                for bucketer_params_combo in itertools.product(*(bucketer_params.values())):
                    for cls_params_combo in itertools.product(*(cls_params.values())):
                        print("Bucketer params are: %s" % str(bucketer_params_combo))
                        print("Cls params are: %s" % str(cls_params_combo))

                        # extract arguments
                        bucketer_args = {'encoding_method': bucket_encoding,
                                         'case_id_col': dataset_manager.case_id_col,
                                         'cat_cols': [dataset_manager.activity_col],
                                         'num_cols': [],
                                         'random_state': random_state}

                        for i in range(len(bucketer_params_names)):
                            bucketer_args[bucketer_params_names[i]] = bucketer_params_combo[i]

                        cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                                            'static_cat_cols': dataset_manager.static_cat_cols,
                                            'static_num_cols': dataset_manager.static_num_cols,
                                            'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                                            'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                                            'fillna': fillna}

                        print(cls_encoder_args)

                        cls_args = {'random_state': random_state,
                                    'min_cases_for_training': n_min_cases_in_bucket}

                        for i in range(len(cls_params_names)):
                            cls_args[cls_params_names[i]] = cls_params_combo[i]

                        # Bucketing prefixes based on control flow
                        print("Bucketing prefixes...")
                        bucketer = get_bucketer(bucket_method, **bucketer_args)
                        bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)

                        pipelines = {}
                        explainers = {}

                        # train and fit pipeline for each bucket
                        count = 0  # storing a few explanations - not all
                        for bucket in set(bucket_assignments_train):

                            print("Fitting pipeline for bucket %s..." % bucket)
                            relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[
                                bucket_assignments_train == bucket]
                            dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                                           relevant_cases_bucket)  # one row per event
                            train_y = dataset_manager.get_label_numeric(dt_train_bucket)

                            feature_combiner = FeatureUnion(
                                [(method, get_encoder(method, **cls_encoder_args)) for method in methods])
                            # TODO: Debug really needed?
                            # pipelines[bucket] = Pipeline([('encoder', feature_combiner), ('debug', Debug()), ('cls', get_classifier(cls_method, **cls_args))])
                            pipelines[bucket] = Pipeline(
                                [('encoder', feature_combiner), ('cls', get_classifier(cls_method, **cls_args))])
                            pipelines[bucket].fit(dt_train_bucket, train_y)

                            # GENERAL EXPLANATIONS

                            print("Generating general explanations...")

                            # SERIALIZE MODEL
                            bucket_path = Path(str(MY_WORKSPACE_DIR) + "/XGBoost/buckets_" + bucket_method + "_" +
                                               cls_encoding + "/train/chunk" + str(part) + "/model_" + dataset_ref
                                               + "_p" + str(part) + "_b" + str(bucket) + "_"
                                               + str(bucketer_params_combo) + "_" + str(cls_params_combo) + ".dat")
                            bucket_path.parent.mkdir(parents=True, exist_ok=True)
                            pickle.dump(pipelines[bucket], open(str(bucket_path.resolve()), "wb"))

                            # get the training data as a matrix
                            trainingdata = feature_combiner.fit_transform(dt_train_bucket)

                            # Did not use categorical features as the parameter - example code of lime says use it,check this out.
                            explainer = lime.lime_tabular.LimeTabularExplainer(trainingdata,
                                                                               feature_names=feature_combiner.get_feature_names(),
                                                                               class_names=['remtime'], verbose=True,
                                                                               mode='regression')
                            print(explainer)

                            # write down feature names

                            feat = pd.DataFrame(feature_combiner.get_feature_names())
                            feature_path = Path(str(MY_WORKSPACE_DIR) + "/XGBoost/buckets_" + bucket_method + "_" + cls_encoding + "/train/chunk" + str(
                                    part) + "/Xtrain_Features_" + dataset_ref + "_p" + str(part) + "_b" + str(
                                    bucket) + ".csv")
                            feature_path.parent.mkdir(parents=True, exist_ok=True)
                            feat.to_csv(str(feature_path.resolve()), index=False)

                            # if the bucketing is prefix-length-based, then evaluate for each prefix length separately, otherwise evaluate all prefixes together
                            max_evaluation_prefix_length = max_prefix_length if bucket_method == "prefix" else min_prefix_length

                            prefix_lengths_test = dt_test_prefixes.groupby(dataset_manager.case_id_col).size()

                            print(max_evaluation_prefix_length)
                            for nr_events in range(min_prefix_length, max_evaluation_prefix_length + 1):
                                gen_counter = gen_counter + 1
                                print("Predicting for %s events..." % nr_events)

                                if bucket_method == "prefix":

                                    # select only prefixes that are of length nr_events
                                    relevant_cases_nr_events = prefix_lengths_test[
                                        prefix_lengths_test == nr_events].index

                                    if len(relevant_cases_nr_events) == 0:
                                        break

                                    dt_test_nr_events = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes,
                                                                                                     relevant_cases_nr_events)
                                    del relevant_cases_nr_events
                                else:
                                    # evaluate on all prefixes
                                    dt_test_nr_events = dt_test_prefixes.copy()

                                start = time()
                                # get predicted cluster for each test case
                                bucket_assignments_test = bucketer.predict(dt_test_nr_events)

                                #### WRITE DOWN TEST RESULTS

                                X_test = pd.DataFrame(dt_test_nr_events)
                                X_path = Path(str(MY_WORKSPACE_DIR) + "/XGBoost/buckets_" + bucket_method + "_" + cls_encoding
                                              + "/test/X_test_" + dataset_ref + "_p" + str(part) + "_e" + str(nr_events) + ".csv")
                                X_path.parent.mkdir(parents=True, exist_ok=True)
                                X_test.to_csv(X_path.resolve())

                                ################################

                                # use the appropriate classifier for each bucket of test cases
                                # for evaluation, collect predictions from different buckets together
                                preds = []
                                test_y = []

                                relevant_cases_bucket = dataset_manager.get_indexes(dt_test_nr_events)[
                                    bucket_assignments_test == bucket]
                                dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_nr_events,
                                                                                              relevant_cases_bucket)  # one row per event

                                if len(relevant_cases_bucket) == 0:
                                    continue

                                elif bucket not in pipelines:
                                    # use mean remaining time (in the training set) as prediction
                                    preds_bucket = array([np.mean(train_chunk["remtime"])] * len(relevant_cases_bucket))
                                    # preds_bucket = [dataset_manager.get_class_ratio(train_chunk)] * len(relevant_cases_bucket)

                                else:
                                    # make actual predictions
                                    preds_bucket = pipelines[bucket].predict_proba(dt_test_bucket)

                                    ####ADDED BY RENUKA - get the explanation
                                    test_y_bucket = dataset_manager.get_label_numeric(dt_test_bucket)
                                    test_x = feature_combiner.fit_transform(dt_test_bucket)[0]

                                    print("Getting classifier...")
                                    # cls = pickle.load(open(MY_WORKSPACE_DIR + "XGBoost/buckets_" + bucket_method + "_" + bucket_encoding + "/train/chunk" + str(part) + "/model_" + dataset_ref + "_p" + str(part) + "_b" + str(bucket) +  "_" + str(bucketer_params_combo) + "_" + str(cls_params_combo) + ".dat", "rb"))
                                    cls = pipelines[bucket].named_steps['cls']

                                    print('Generating local Explanations')
                                    exp = generate_local_explanations(explainer, test_x, cls, test_y_bucket,
                                                                      feature_combiner)

                                    exp_dict[exp_indx] = exp
                                    exp_indx = exp_indx + 1

                                    # rc={'axes.labelsize': 12, 'xtick.labelsize': 13, 'ytick.labelsize': 13 , 'axes.titlesize': 10}
                                    # sns.set(rc)
                                    # sns.set_style("whitegrid")
                                    # %matplotlib inline

                                    print('Explanations for prefix length ', bucket)

                                    # fig = exp.as_pyplot_figure()

                                    # print("\nSAVING FIGURE...\n")
                                    # gen_counter = gen_counter+1
                                    # print(MY_WORKSPACE_DIR + "/XGBoost/buckets_" + bucket_method + "_" + cls_encoding + "/lime/chunk" + str(part) + "/Local_Expl_" + dataset_ref + "_p" + str(part) + "_e" + str(nr_events) + "_b" + str(bucket) +"_" + str(bucketer_params_combo) + "_" + str(cls_params_combo) + ".png")

                                    # fig.savefig(MY_WORKSPACE_DIR + "/XGBoost/buckets_" + bucket_method + "_" + cls_encoding + "/lime/chunk" + str(part) + "/Local_Expl_" + dataset_ref+ "_G" + str(gen_counter) + "_p" + str(part) + "_e" + str(nr_events) + "_b" + str(bucket) + ".png",
                                    #           bbox_inches='tight',dpi=300)
                                    # gen_counter = gen_counter+1

                                    count = count + 1
                                    ##############

                            preds_bucket = preds_bucket.clip(
                                min=0)  # if remaining time is predicted to be negative, make it zero
                            preds.extend(preds_bucket)

                            # extract actual label values
                            test_y_bucket = dataset_manager.get_label_numeric(dt_test_bucket)  # one row per case
                            test_y.extend(test_y_bucket)

                            ##### WRITE DOWN RESUTLS
                            y_test = pd.DataFrame(test_y)
                            y_path = Path(str(MY_WORKSPACE_DIR) + "/XGBoost/buckets_" + bucket_method + "_" + cls_encoding + "/test/y_test_" + dataset_ref + "_p" + str(
                                    part) + ".csv")
                            y_test.to_csv(str(y_path.resolve()))
                            ##########################

                        if len(test_y) < 2:
                            mae = None
                        else:
                            mae = mean_absolute_error(test_y, preds)

                        # prec, rec, fscore, _ = precision_recall_fscore_support(test_y, [0 if pred < 0.5 else 1 for pred in preds], average="binary")
                        bucketer_params_str = ";".join([str(param) for param in bucketer_params_combo])
                        cls_params_str = ";".join([str(param) for param in cls_params_combo])

                        print([part, dataset_name, method_name, cls_method, bucketer_params_str, cls_params_str,
                               nr_events,
                               "mae", mae])
                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (
                            part, dataset_name, method_name, cls_method, bucketer_params_str, cls_params_str, nr_events,
                            "mae",
                            mae))
                        # fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, cls_method, bucketer_params_str, cls_params_str, nr_events, "precision", prec))
                        # fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, cls_method, bucketer_params_str, cls_params_str, nr_events, "recall", rec))
                        # fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, cls_method, bucketer_params_str, cls_params_str, nr_events, "fscore", fscore))

                    print("\n")

    # LIME Explanations

    rc = {'axes.labelsize': 12, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'axes.titlesize': 10}
    sns.set(rc)
    sns.set_style("whitegrid")
    for pre, exp in exp_dict.items():
        print('Explanations for prefix length ', bucket)
        fig = exp.as_pyplot_figure()
        fig_path = Path(str(MY_WORKSPACE_DIR) + "/XGBoost/buckets_" + bucket_method + "_" + cls_encoding + "/lime/chunk" + str(
            part) + "/Local_Expl_" + dataset_ref + "_p" + str(pre) + "_b" + str(bucket) + ".png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(fig_path.resolve()), bbox_inches='tight',
                    dpi=300)
        fig.show()
        # TODO: Hack to stop exceptions of Too many http requests being thrown
        sleep(5)


if __name__ == "__main__":
    train()

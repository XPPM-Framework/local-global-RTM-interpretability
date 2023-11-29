import pandas as pd
import numpy as np
from numpy import array

import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from xgboost.core import XGBoostError

MY_WORKSPACE_DIR = "./"

# #### Classifiers

# ##### Classifier Wrapper

# In[64]:


class ClassifierWrapper(object):

    def __init__(self, cls, min_cases_for_training = 30):
        self.cls = cls

        self.min_cases_for_training = min_cases_for_training
        self.hardcoded_prediction = None

    def fit(self, X, y):
        # if there are too few training instances, use the mean
        if X.shape[0] < self.min_cases_for_training:
            self.hardcoded_prediction = np.mean(y)

        # if all the training instances are of the same class, use this class as prediction
        elif len(set(y)) < 2:
            self.hardcoded_prediction = int(y[0])

        else:
            self.cls.fit(X, y)
            return self

    def predict_proba(self, X, y=None):
        if self.hardcoded_prediction is not None:
            return array([self.hardcoded_prediction] * X.shape[0])

        else:
            #preds_pos_label_idx = np.where(self.cls.classes_ == 0)[0][0]
            #preds = self.cls.predict_proba(X)[:,preds_pos_label_idx]
            preds = self.cls.predict(X)
            return preds

    def fit_predict(self, X, y):

        self.fit(X, y)
        return self.predict_proba(X)


# ##### Classifier Factory

# In[65]:


def get_classifier(method, n_estimators, max_features=None, learning_rate=None, max_depth=None, random_state=None, subsample=None, colsample_bytree=None, min_cases_for_training=30):

    if method == "rf":
        return ClassifierWrapper(
            cls=RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=random_state),
            min_cases_for_training=min_cases_for_training)

    elif method == "xgb":
        return ClassifierWrapper(
            cls=xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample,
                                 max_depth=max_depth, colsample_bytree=colsample_bytree, n_jobs=2),
            min_cases_for_training=min_cases_for_training)

    else:
        print("Invalid classifier type")
        return None


# Adding LIME utility function
#

# In[68]:


import seaborn as sns
import matplotlib.pyplot as plt

def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
        .sort_values('feature_importance', ascending = False) \
        .reset_index(drop = True)
    return df

# plotting a feature importance dataframe (horizontal barchart)
def var_imp_plot(imp_df, title, num_feat, bucket_method, bucket_encoding, part, dataset_ref, gen_counter, nr_events, bucket, bucketer_params_combo, cls_params_combo):

    try:
        imp_df.columns = ['feature', 'feature_importance']
        #plt.figure( figsize=(25,10))
        b= sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df.head(num_feat), orient = 'h', palette="Blues_r")
        b.set_title(title, fontsize = 14)

        for item in b.get_yticklabels():
            item.set_fontsize(13)

        #%get_backend()
        fig2 = b.get_figure()

        print("\nSAVING FIGURE\n")

        print(MY_WORKSPACE_DIR + "XGBoost/buckets_" + bucket_method + "_" + bucket_encoding + "/lime/chunk" + str(part) + "/FeatureImportance_" + dataset_ref + "_G" + str(gen_counter)+ "_p" + str(part) + "_e" + str(nr_events)+ "_b" + str(bucket) +"_" + str(bucketer_params_combo) + "_" + str(cls_params_combo)+ ".png")
        fig2.show()
        fig2.savefig(MY_WORKSPACE_DIR + "XGBoost/buckets_" + bucket_method + "_" + bucket_encoding + "/lime/chunk" + str(part) + "/FeatureImportance_" + dataset_ref + "_G" + str(gen_counter)+"_p" + str(part) + "_e" + str(nr_events)+ "_b" + str(bucket) +"_" + str(bucketer_params_combo) + "_" + str(cls_params_combo)+ ".png",
                     bbox_inches='tight',dpi=300)
    except XGBoostError:
        print("############################")
        print("XGBOOST EXCEPTION")
        print("############################")



# In[69]:


def generate_local_explanations(explainer,test_xi, cls,test_y, num_vars = 6):

    print("Actual value ", test_y)
    num_features=6;# maximum is 6 ,if it is larger than 6, the features displayed are different.

    exp = None
    try:
        exp = explainer.explain_instance(test_xi, cls.predict_proba, num_features=num_features)
        exp.show_in_notebook(show_table=True, show_all=False)
        print ('Explanation for class %s' )
        print ('\n'.join(map(str, exp.as_list())))

    except ValueError:
        print("#################################")
        print("EXCEPTION")
        print("#################################")

    #probability_result=cls.predict_proba([test_xi])[0];
    #print(probability_result);

    return exp

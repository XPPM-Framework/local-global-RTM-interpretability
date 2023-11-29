import pandas as pd

MY_WORKSPACE_DIR = "./"

# trying to get the datset out of the pipeline process
from sklearn.base import TransformerMixin, BaseEstimator

class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        #self.shape = shape

        # what other output you want
        return X

    def fit(self, X, y=None, **fit_params):

        X_train = pd.DataFrame( X )
        y_train = pd.DataFrame( y )

        #print("[DEBUG] Writing file: " + MY_WORKSPACE_DIR + "XGBoost/buckets_" + bucket_method + "_" + cls_encoding + "/train/chunk" + str(part) + "/Xtrain_" + dataset_ref + "_p" + str(part) + "_b" + str(bucket) + ".csv")
        #X_train.to_csv( MY_WORKSPACE_DIR + "XGBoost/buckets_" + bucket_method + "_" + cls_encoding + "/train/chunk" + str(part) + "/Xtrain_" + dataset_ref + "_p" + str(part) + "_b" + str(bucket) + ".csv", index=False)
        #y_train.to_csv( MY_WORKSPACE_DIR + "XGBoost/buckets_" + bucket_method + "_" + cls_encoding + "/train/chunk"  + str(part) + "/ytrain_" + dataset_ref  + "_p" + str(part) + "_b" + str(bucket) + ".csv", index=False)

        return self



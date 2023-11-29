import pandas as pd
import numpy as np
from time import time

from sklearn.cluster import KMeans
from sklearn.base import TransformerMixin

from transformers import StaticTransformer, LastStateTransformer, AggregateTransformer, IndexBasedTransformer


def get_bucketer(method, encoding_method=None, case_id_col=None, cat_cols=None, num_cols=None, n_clusters=None,
                 random_state=None, n_neighbors=None):
    if method == "cluster":
        bucket_encoder = get_encoder(method=encoding_method, case_id_col=case_id_col, dynamic_cat_cols=cat_cols,
                                     dynamic_num_cols=num_cols)
        clustering = KMeans(n_clusters, random_state=random_state)
        return ClusterBasedBucketer(encoder=bucket_encoder, clustering=clustering)

    elif method == "state":
        bucket_encoder = get_encoder(method=encoding_method, case_id_col=case_id_col, dynamic_cat_cols=cat_cols,
                                     dynamic_num_cols=num_cols)
        return StateBasedBucketer(encoder=bucket_encoder)

    elif method == "single":
        return NoBucketer(case_id_col=case_id_col)

    elif method == "prefix":
        return PrefixLengthBucketer(case_id_col=case_id_col)

    elif method == "knn":
        bucket_encoder = get_encoder(method=encoding_method, case_id_col=case_id_col, dynamic_cat_cols=cat_cols,
                                     dynamic_num_cols=num_cols)
        return KNNBucketer(encoder=bucket_encoder, n_neighbors=n_neighbors)

    else:
        print("Invalid bucketer type")
        return None


def get_encoder(method, case_id_col=None, static_cat_cols=None, static_num_cols=None, dynamic_cat_cols=None,
                dynamic_num_cols=None, fillna=True, max_events=None):
    if method == "static":
        return StaticTransformer(case_id_col=case_id_col, cat_cols=static_cat_cols, num_cols=static_num_cols,
                                 fillna=fillna)

    elif method == "last":
        return LastStateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols,
                                    fillna=fillna)

    elif method == "agg":
        return AggregateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols,
                                    boolean=False, fillna=fillna)

    elif method == "bool":
        return AggregateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols,
                                    boolean=True, fillna=fillna)

    elif method == "index":
        return IndexBasedTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols,
                                     max_events=max_events, fillna=fillna)

    else:
        print("Invalid encoder type")
        return None


# ##### Zero bucketing
#
# All prefix traces are considered to be ub the same bucket. As such, a single predictor is fit for all prefies in the prefix log.

# In[47]:


# All prefix traces are considered to be in the same bucket. As such, a single
# predictor is fit for all prefies in the prefix log.
class NoBucketer(object):

    def __init__(self, case_id_col):
        self.n_states = 1
        self.case_id_col = case_id_col

    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        return np.ones(len(X[self.case_id_col].unique()), dtype=np.int)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)


# ##### Prefix length bucketing
#
# Each bucket contains the prefixes of a specific length. For instance, the *nth* bucket contains prefixes where at least *n* events have been performed. **One classifier is built for each possible prefix length**.
#

# In[48]:


# Prefix length bucketing. Each bucket contains the prefixes of a specific
# length. For instance, the nth bucket contains prefixes where at least n events
# have been performed. One classifier is built for each possible prefix length.
class PrefixLengthBucketer(object):

    def __init__(self, case_id_col):
        self.n_states = 0
        self.case_id_col = case_id_col

    def fit(self, X, y=None):
        sizes = X.groupby(self.case_id_col).size()
        self.n_states = sizes.unique()
        return self

    def predict(self, X, y=None):
        return X.groupby(self.case_id_col).size().to_numpy()

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)


#
#  ##### Cluster bucketing
#
#  Each bucket represents a cluster that results from applying a clustering algorithmon the encoded prefixes. One classifier is trained for each resulting cluster, considering only the historical prefixes that fall into that particular cluster. At runtime, the cluster of the running case is determined based on its similarity to each of the existing clusters and the corresponding classifier is applied.
#

# In[49]:


# Each bucket represents a cluster that results from applying a clustering
# algorithmon the encoded prefixes. One classifier is trained for each
# resulting cluster, considering only the historical prefixes that fall into that
# particular cluster. At runtime, the cluster of the running case is determined
# based on its similarity to each of the existing clusters and the corresponding
# classifier is applied.
class ClusterBasedBucketer(object):

    def __init__(self, encoder, clustering):
        self.encoder = encoder
        self.clustering = clustering

    def fit(self, X, y=None):
        dt_encoded = self.encoder.fit_transform(X)
        self.clustering.fit(dt_encoded)
        return self

    def predict(self, X, y=None):
        dt_encoded = self.encoder.transform(X)
        return self.clustering.predict(dt_encoded)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)


# ##### State Bucketing
#
# It is used in process-aware apporaches where some kind of process representation is derived and a predictor is trained for each state, or dedcision point.

# In[50]:


# It is used in process-aware apporaches where some kind of process representation
# is derived and a predictor is trained for each state, or dedcision point.
class StateBasedBucketer(object):

    def __init__(self, encoder):
        self.encoder = encoder
        self.dt_states = None
        self.n_states = 0

    def fit(self, X, y=None):
        dt_encoded = self.encoder.fit_transform(X)
        self.dt_states = dt_encoded.drop_duplicates()
        self.dt_states = self.dt_states.assign(state=range(len(self.dt_states)))
        self.n_states = len(self.dt_states)
        return self

    def predict(self, X, y=None):
        dt_encoded = self.encoder.transform(X)
        dt_transformed = pd.merge(dt_encoded, self.dt_states, how='left')
        dt_transformed.fillna(-1, inplace=True)
        return dt_transformed["state"].astype(int).as_matrix()

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)

# ##### Aggregate Transformer

# In[53]:


# Probably not actually used because it was overwritten by different declaration in separate cell
class AggregateTransformer_Unused(TransformerMixin):

    def __init__(self, case_id_col, cat_cols, num_cols, boolean=False, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols

        self.boolean = boolean
        self.fillna = fillna

        self.columns = None

        self.fit_time = 0
        self.transform_time = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        start = time()

        # transform numeric cols
        if len(self.num_cols) > 0:
            dt_numeric = X.groupby(self.case_id_col)[self.num_cols].agg(["mean", "max", "min", "sum", "std"])
            dt_numeric.columns = ['_'.join(col).strip() for col in dt_numeric.columns]

        # transform cat cols
        print(X)
        print("#########################")
        dt_transformed = pd.get_dummies(X[self.cat_cols])
        print(dt_transformed)
        print("#########################")
        dt_transformed[self.case_id_col] = X[self.case_id_col]
        print(dt_transformed)
        print("##########################")
        del X
        if self.boolean:
            dt_transformed = dt_transformed.groupby(self.case_id_col).max()
        else:
            dt_transformed = dt_transformed.groupby(self.case_id_col).sum()

        print(dt_transformed)
        print("##########################")
        # concatenate
        if len(self.num_cols) > 0:
            dt_transformed = pd.concat([dt_transformed, dt_numeric], axis=1)
            del dt_numeric

        # fill missing values with 0-s
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)

        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]

        self.transform_time = time() - start
        return dt_transformed


# ##### Index Based Transformer

# In[54]:


class IndexBasedExtractor(TransformerMixin):

    def __init__(self, cat_cols, num_cols, max_events, fillna=True):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.max_events = max_events
        self.fillna = fillna
        self.columns = None

        self.fit_time = 0
        self.transform_time = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        start = time()

        # add missing columns if necessary
        if self.columns is None:
            relevant_num_cols = ["%s_%s" % (col, i) for col in self.num_cols for i in range(self.max_events)]
            relevant_cat_col_prefixes = tuple(
                ["%s_%s_" % (col, i) for col in self.cat_cols for i in range(self.max_events)])
            relevant_cols = [col for col in X.columns if col.startswith(relevant_cat_col_prefixes)] + relevant_num_cols
            self.columns = relevant_cols
        else:
            missing_cols = [col for col in self.columns if col not in X.columns]
            for col in missing_cols:
                X[col] = 0

        self.transform_time = time() - start
        return X[self.columns]


# Probably not actually used because it was overwritten by different declaration in separate cell
class IndexBasedTransformer_Unused(TransformerMixin):

    def __init__(self, case_id_col, cat_cols, num_cols, max_events=None, fillna=True, create_dummies=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.max_events = max_events
        self.fillna = fillna
        self.create_dummies = create_dummies

        self.columns = None

        self.fit_time = 0
        self.transform_time = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        start = time()

        grouped = X.groupby(self.case_id_col, as_index=False)

        if self.max_events is None:
            self.max_events = grouped.size().max()

        dt_transformed = pd.DataFrame(grouped.apply(lambda x: x.name), columns=[self.case_id_col])
        for i in range(self.max_events):
            dt_index = grouped.nth(i)[[self.case_id_col] + self.cat_cols + self.num_cols]
            dt_index.columns = [self.case_id_col] + ["%s_%s" % (col, i) for col in self.cat_cols] + ["%s_%s" % (col, i)
                                                                                                     for col in
                                                                                                     self.num_cols]
            dt_transformed = pd.merge(dt_transformed, dt_index, on=self.case_id_col, how="left")
        dt_transformed.index = dt_transformed[self.case_id_col]

        # one-hot-encode cat cols
        if self.create_dummies:
            all_cat_cols = ["%s_%s" % (col, i) for col in self.cat_cols for i in range(self.max_events)]
            dt_transformed = pd.get_dummies(dt_transformed, columns=all_cat_cols).drop(self.case_id_col, axis=1)

        # fill missing values with 0-s
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)

        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]

        self.transform_time = time() - start
        return dt_transformed

# ##### Previous State Transformer

# In[56]:


class PreviousStateTransformer(TransformerMixin):

    def __init__(self, case_id_col, cat_cols, num_cols, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna

        self.columns = None

        self.fit_time = 0
        self.transform_time = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        start = time()

        dt_last = X.groupby(self.case_id_col).nth(-2)

        # transform numeric cols
        dt_transformed = dt_last[self.num_cols]

        # transform cat cols
        if len(self.cat_cols) > 0:
            dt_cat = pd.get_dummies(dt_last[self.cat_cols])
            dt_transformed = pd.concat([dt_transformed, dt_cat], axis=1)

        # add 0 rows where previous value did not exist
        dt_transformed = dt_transformed.reindex(X.groupby(self.case_id_col).first().index, fill_value=0)

        # fill NA with 0 if requested
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)

        # add missing columns if necessary
        if self.columns is not None:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
        else:
            self.columns = dt_transformed.columns

        self.transform_time = time() - start
        return dt_transformed

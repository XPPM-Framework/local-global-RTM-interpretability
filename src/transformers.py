from time import time
import pandas as pd

from sklearn.base import TransformerMixin


# #### Transformers
# ##### Aggregate Transformer

# In[58]:


class AggregateTransformer(TransformerMixin):

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

    def get_feature_names(self):
        return self.columns.tolist()

    def transform(self, X, y=None):
        start = time()

        # transform numeric cols
        if len(self.num_cols) > 0:
            dt_numeric = X.groupby(self.case_id_col)[self.num_cols].agg(["mean", "max", "min", "sum", "std"])
            dt_numeric.columns = ['_'.join(col).strip() for col in dt_numeric.columns]

        # transform cat cols
        dt_transformed = pd.get_dummies(X[self.cat_cols])
        dt_transformed[self.case_id_col] = X[self.case_id_col]
        del X
        if self.boolean:
            dt_transformed = dt_transformed.groupby(self.case_id_col).max()
        else:
            dt_transformed = dt_transformed.groupby(self.case_id_col).sum()

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


# ##### Index Based Extractor

# In[59]:


class IndexBasedExtractor(TransformerMixin):

    def __init__(self, cat_cols, num_cols, max_events, fillna=True):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.max_events = max_events
        self.fillna = fillna
        self.columns = None

        self.fit_time = 0
        self.transform_time = 0

    def get_feature_names(self):
        return self.columns.tolist()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        start = time()

        # add missing columns if necessary
        if self.columns is None:
            relevant_num_cols = ["%s_%s"%(col, i) for col in self.num_cols for i in range(self.max_events)]
            relevant_cat_col_prefixes = tuple(["%s_%s_"%(col, i) for col in self.cat_cols for i in range(self.max_events)])
            relevant_cols = [col for col in X.columns if col.startswith(relevant_cat_col_prefixes)] + relevant_num_cols
            self.columns = relevant_cols
        else:
            missing_cols = [col for col in self.columns if col not in X.columns]
            for col in missing_cols:
                X[col] = 0

        self.transform_time = time() - start
        return X[self.columns]


# ##### Index Based Transformer

# In[60]:


class IndexBasedTransformer(TransformerMixin):

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

    def get_feature_names(self):
        return self.columns.tolist()

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
            dt_index.columns = [self.case_id_col] + ["%s_%s"%(col, i) for col in self.cat_cols] + ["%s_%s"%(col, i) for col in self.num_cols]
            dt_transformed = pd.merge(dt_transformed, dt_index, on=self.case_id_col, how="left")
        dt_transformed.index = dt_transformed[self.case_id_col]

        # one-hot-encode cat cols
        if self.create_dummies:
            all_cat_cols = ["%s_%s"%(col, i) for col in self.cat_cols for i in range(self.max_events)]
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


# ##### Last State Transformer

# In[61]:


class LastStateTransformer(TransformerMixin):

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

    def get_feature_names(self):
        return self.columns.tolist()

    def transform(self, X, y=None):
        start = time()

        dt_last = X.groupby(self.case_id_col).last()

        # transform numeric cols
        dt_transformed = dt_last[self.num_cols]

        # transform cat cols
        if len(self.cat_cols) > 0:
            dt_cat = pd.get_dummies(dt_last[self.cat_cols])
            dt_transformed = pd.concat([dt_transformed, dt_cat], axis=1)

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


# ##### Previous State Transformer

# In[62]:


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

    def get_feature_names(self):
        return self.columns.tolist()

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



# ##### Static Transformer

# In[63]:


class StaticTransformer(TransformerMixin):

    def __init__(self, case_id_col, cat_cols, num_cols, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.fillna = fillna

        self.columns = None

        self.fit_time = 0
        self.transform_time = 0

    def get_feature_names(self):
        return self.columns.tolist()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        start = time()

        dt_first = X.groupby(self.case_id_col).first()

        # transform numeric cols
        dt_transformed = dt_first[self.num_cols]

        # transform cat cols
        if len(self.cat_cols) > 0:
            dt_cat = pd.get_dummies(dt_first[self.cat_cols])
            dt_transformed = pd.concat([dt_transformed, dt_cat], axis=1)

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



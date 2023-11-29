import pandas as pd
import numpy as np

from sklearn.model_selection import KFold

MY_WORKSPACE_DIR = "./"
# #### Dataset Manager

# ##### Dataset Configurations

# In[66]:

case_id_col = {}
activity_col = {}
timestamp_col = {}
label_col = {}
pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
filename = {}

################################################################################
#                        BPIC2011 settings                                     #
################################################################################
dataset = "bpic2011"
filename[dataset] = "logdata/bpic2011.csv"

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "remtime"
pos_label[dataset] = "deviant"
neg_label[dataset] = "regular"

# features for classifier
dynamic_cat_cols[dataset] = ["Activity", "Producer code", "Section", "Specialism code", "group"]
static_cat_cols[dataset] = ["Diagnosis", "Treatment code", "Diagnosis code", "case Specialism code", "Diagnosis Treatment Combination ID"]
dynamic_num_cols[dataset] = ["Number of executions", "duration", "month", "weekday", "hour"]
static_num_cols[dataset] = ["Age"]

################################################################################
#                        BPIC2015 settings                                     #
################################################################################
dataset = "bpic2015_5"
filename[dataset] = "logdata/bpic2015_5.csv"

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "remtime"
pos_label[dataset] = "deviant"
neg_label[dataset] = "regular"

# features for classifier
dynamic_cat_cols[dataset] = ["Activity", "monitoringResource", "question", "Resource"]
static_cat_cols[dataset] = ["Responsible_actor"]
dynamic_num_cols[dataset] = ["duration", "month", "weekday", "hour"]
static_num_cols[dataset] = ["SUMleges", 'Aanleg (Uitvoeren werk of werkzaamheid)', 'Bouw', 'Brandveilig gebruik (vergunning)', 'Gebiedsbescherming', 'Handelen in strijd met regels RO', 'Inrit/Uitweg', 'Kap', 'Milieu (neutraal wijziging)', 'Milieu (omgevingsvergunning beperkte milieutoets)', 'Milieu (vergunning)', 'Monument', 'Reclame', 'Sloop']
static_num_cols[dataset].append('Flora en Fauna')
static_num_cols[dataset].append('Brandveilig gebruik (melding)')
static_num_cols[dataset].append('Milieu (melding)')

################################################################################
#                        BPIC2012 A settings                                   #
################################################################################
dataset = "bpic2012a"

filename[dataset] = "logdata/bpic2012a.csv"

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "remtime"
pos_label[dataset] = "regular"
neg_label[dataset] = "deviant"

# features for classifier
dynamic_cat_cols[dataset] = ['Activity', 'Resource']
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = ['open_cases','elapsed']
static_num_cols[dataset] = ['AMOUNT_REQ']

################################################################################
#                        BPIC2012 O settings                                   #
################################################################################
dataset = "bpic2012o"

filename[dataset] = "logdata/bpic2012o.csv"

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "remtime"
pos_label[dataset] = "regular"
neg_label[dataset] = "deviant"

# features for classifier
dynamic_cat_cols[dataset] = ['Activity', 'Resource']
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = ['open_cases','elapsed']
static_num_cols[dataset] = ['AMOUNT_REQ']

################################################################################
#                        BPIC2012 W settings                                   #
################################################################################
dataset = "bpic2012w"

filename[dataset] = "logdata/bpic2012w.csv.calc"

case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity"
timestamp_col[dataset] = "Complete Timestamp"
label_col[dataset] = "remtime"
pos_label[dataset] = "regular"
neg_label[dataset] = "deviant"

# features for classifier
dynamic_cat_cols[dataset] = ['Activity', 'Resource']
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = ['open_cases','elapsed',]
static_num_cols[dataset] = ['AMOUNT_REQ']

# ##### Dataset Manager

# In[67]:


class DatasetManager:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        self.case_id_col = case_id_col[self.dataset_name]
        self.activity_col = activity_col[self.dataset_name]
        self.timestamp_col = timestamp_col[self.dataset_name]
        self.label_col = label_col[self.dataset_name]
        self.pos_label = pos_label[self.dataset_name]

        self.dynamic_cat_cols = dynamic_cat_cols[self.dataset_name]
        self.static_cat_cols = static_cat_cols[self.dataset_name]
        self.dynamic_num_cols = dynamic_num_cols[self.dataset_name]
        self.static_num_cols = static_num_cols[self.dataset_name]

    def read_dataset(self):
        # read dataset
        dtypes = {col:"object" for col in self.dynamic_cat_cols+self.static_cat_cols+[self.case_id_col, self.timestamp_col]}
        for col in self.dynamic_num_cols + self.static_num_cols:
            dtypes[col] = "float"

        dtypes[self.label_col] = "float"  # remaining time should be float

        data = pd.read_csv( MY_WORKSPACE_DIR + "experiments/" + filename[ self.dataset_name], sep=",", dtype=dtypes)
        data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])
        return data

    def split_data(self, data, train_ratio):
        # split into train and test using temporal split

        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        return (train, test)

    def generate_prefix_data(self, data, min_length, max_length):
        # generate prefix data (each possible prefix becomes a trace)
        data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

        dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length)
        for nr_events in range(min_length+1, max_length+1):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)

        dt_prefixes['case_length'] = dt_prefixes.groupby(self.case_id_col)[self.activity_col].transform(len)
        return dt_prefixes

    def get_pos_case_length_quantile(self, data, quantile=0.90):
        return int(np.floor(data.groupby(self.case_id_col).size().quantile(quantile)))

    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_label(self, data):
        return data.groupby(self.case_id_col).min()[self.label_col]

    def get_label_numeric(self, data):
        y = self.get_label(data) # one row per case
        #return [1 if label == self.pos_label else 0 for label in y]
        return y

    def get_class_ratio(self, data):
        class_freqs = data[self.label_col].value_counts()
        return class_freqs[self.pos_label] / class_freqs.sum()

    def get_stratified_split_generator(self, data, n_splits=5, shuffle=True, random_state=22):
        grouped_firsts = data.groupby(self.case_id_col, as_index=False).first()
        skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        for train_index, test_index in skf.split(grouped_firsts, grouped_firsts[self.label_col]):
            current_train_names = grouped_firsts[self.case_id_col][train_index]
            train_chunk = data[data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            test_chunk = data[~data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            yield (train_chunk, test_chunk)



import pandas as pd
import numpy as np

# #### Preprocessing
#
# Adding columns:
#   - elapsed time
#   - remaining time


from env_vars import MY_WORKSPACE_DIR


filenames_bpic2011 = "bpic2011.csv"

filenames_bpic2012a = "bpic2012a.csv"
filenames_bpic2012o = "bpic2012o.csv"
filenames_bpic2012w = "bpic2012w.csv"

filenames_bpic2015 = "bpic2015_5.csv"

filenames = [filenames_bpic2011, filenames_bpic2012a, filenames_bpic2012o, filenames_bpic2012w, filenames_bpic2015]

columns_to_remove = ["label"]

import datetime


def add_remtime_column(group, timestamp_col="Complete Timestamp"):
    group = group.sort_values(timestamp_col, ascending=False)
    start_date = group[timestamp_col].iloc[-1]
    end_date = group[timestamp_col].iloc[0]

    elapsed = group[timestamp_col] - start_date
    elapsed = elapsed.fillna(datetime.timedelta(0))
    group["elapsed"] = elapsed.apply(lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds

    remtime = end_date - group[timestamp_col]
    remtime = remtime.fillna(datetime.timedelta(0))
    group["remtime"] = remtime.apply(lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds

    return group


def get_open_cases(event_log: pd.DataFrame, case_id_key: str, timestamp_key: str) -> pd.Series:
    """
    Get running cases for each event. Assumes all cases to be completed.
    :param event_log: The event log to get the open cases metric for.
    :param case_id_key: The case id key of the event log.
    :param timestamp_key: The timestamp key of the event log.
    :return:
    """
    # Find start and end for each case
    event_log = event_log.sort_values(timestamp_key, ascending=True,
                                      kind="mergesort")  # Mapping of case_id to (start, end)
    start_end_df = event_log.groupby(case_id_key).agg(
        start=pd.NamedAgg(column=timestamp_key, aggfunc="first"),
        end=pd.NamedAgg(column=timestamp_key, aggfunc="last"),
    )

    def calculate_open_cases(row):
        timestamp = row[timestamp_key]
        open_case_count = ((start_end_df['start'] <= timestamp) & (timestamp <= start_end_df['end'])).sum()
        return open_case_count

    open_cases = event_log.apply(calculate_open_cases, axis=1)

    return open_cases

input_data_folder = MY_WORKSPACE_DIR / "experiments/main_logs/"
output_data_folder = MY_WORKSPACE_DIR / "experiments/logdata/"
for filename in filenames:
    print(filename)
    # data = pd.read_csv(os.path.join(input_data_folder, filename), sep=",")
    # data = data.drop([columns_to_remove], axis=1)
    # data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    # data = data.groupby(case_id_col).apply(add_remtime_column)
    # data.to_csv(os.path.join(output_data_folder, filename), sep=";", index=False)

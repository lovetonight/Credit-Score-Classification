import pandas as pd

def transform_x52(value):
    return -value

def get_data(flag = False):
    df = pd.read_csv('../data/full_data.csv')
    if flag is True:
        df['x52'] = df['x52'].apply(transform_x52)
    addresses_ignore = []
    filtered_df = df[~df["address"].isin(addresses_ignore)]
    addresses = filtered_df["address"].values
    X = filtered_df[
        [
            "x11",
            "x12",
            "x21",
            "x22",
            "x23",
            "x24",
            "x31",
            "x32",
            "x41",
            "x42",
            "x43",
            "x51",
            "x52",
            "x61",
            "x71",
            "x72",
        ]
    ].values
    y = filtered_df["label"].values
    return X, y



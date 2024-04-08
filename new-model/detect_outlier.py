from datetime import datetime
import pandas as pd
import numpy as np


def read_data():
    df = pd.read_csv("./data/all_data.csv")
    df["borrow_per_balance"] = np.where(
        df["balanceInUSD"] == 0, 0, df["borrowInUSD"] / df["balanceInUSD"]
    )
    df["deposit_per_balance"] = np.where(
        df["balanceInUSD"] == 0, 0, df["depositInUSD"] / df["balanceInUSD"]
    )
    df["borrow_per_deposit"] = np.where(
        df["depositInUSD"] == 0, 0, df["borrowInUSD"] / df["depositInUSD"]
    )
    # df["avg_liquidation"] = np.where(
    #     (df["totalValueOfLiquidation"] == 0) | (df["numberOfLiquidation"] == 0),
    #     0,
    #     df["totalValueOfLiquidation"] / df["numberOfLiquidation"],
    # )
    current_timestamp = datetime.now()
    df["createdAt"] = pd.to_datetime(df["createdAt"])
    df["createdAt"] = current_timestamp - df["createdAt"]
    min_value = df['createdAt'].min()
    max_value = df['createdAt'].max()
    df = df.copy()
    df['age'] = (df['createdAt'] - min_value) * (1000 - 0) / (max_value - min_value) + 0

    # Drop column
    df = (
        df.drop("_id", axis=1)
        .drop("depositInUSD", axis=1)
        .drop("borrowInUSD", axis=1)
        .drop("numberCallsTopContracts", axis=1)
        # .drop("avg_liquidation", axis=1)
    )
    return df


def read_data_no_outlier():
    df = read_data()
    df = df.dropna()
    # Balance outlier
    # Sử dụng z-score 1
    label_value_in_usd = [
        "balanceInUSD",
        "totalValueOfLiquidation",
        "averageTotalAsset",
        "frequencyMountOfTransaction",
        "borrow_per_balance",
        "deposit_per_balance",
        "borrow_per_deposit",
    ]
    for label in label_value_in_usd:
        df["zscore"] = (df[label] - df[label].mean()) / df[label].std()
        df = df[df["zscore"] < 1]
    return df
    # Number of - outlier

read_data_no_outlier()

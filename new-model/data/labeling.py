import pandas as pd

file_path = "all_data_10_5.csv"
df = pd.read_csv(file_path)


def add_label(df):
    for index, row in df.iterrows():
        balanceInUSD = row["balanceInUSD"]
        numberOfLiquidation = row["numberOfLiquidation"]
        borrowInUSD = row["borrowInUSD"]
        # Label 0
        if balanceInUSD <= 1000 and numberOfLiquidation > 0:
            df.loc[index, "1st_label"] = 0
            df.loc[index, "2nd_label"] = 0
        if 1000 < balanceInUSD <= 100000 and borrowInUSD / balanceInUSD >= 0.6 and numberOfLiquidation > 0:
            df.loc[index, "1st_label"] = 0
            df.loc[index, "2nd_label"] = 1
        if (
            balanceInUSD > 0
            and borrowInUSD / balanceInUSD >= 0.4
            and numberOfLiquidation > 0
        ):
            df.loc[index, "1st_label"] = 0
            df.loc[index, "2nd_label"] = 1

        # Label 1
        if balanceInUSD <= 1000 and numberOfLiquidation == 0:
            df.loc[index, "1st_label"] = 1
            df.loc[index, "2nd_label"] = 1
        if (
            0 < balanceInUSD <= 100000
            and 0.4 <= borrowInUSD / balanceInUSD < 0.6
            and numberOfLiquidation == 0
        ):
            df.loc[index, "1st_label"] = 1
            df.loc[index, "2nd_label"] = 2
        if (
            balanceInUSD > 1000
            and numberOfLiquidation > 0
            and 0 < borrowInUSD / balanceInUSD < 0.4
        ):
            df.loc[index, "1st_label"] = 1
            df.loc[index, "2nd_label"] = 0

        # Label 2
        if 1000 < balanceInUSD < 10000 and numberOfLiquidation == 0:
            df.loc[index, "1st_label"] = 2
            df.loc[index, "2nd_label"] = 3
        if (
            balanceInUSD > 1000
            and 0.3 <= borrowInUSD / balanceInUSD < 0.4
            and numberOfLiquidation == 0
        ):
            df.loc[index, "1st_label"] = 2
            df.loc[index, "2nd_label"] = 2

        # Label 3
        if (
            balanceInUSD > 1000000
            and numberOfLiquidation == 0
            and 0.2 <= borrowInUSD / balanceInUSD < 0.3
        ):
            df.loc[index, "1st_label"] = 3
            df.loc[index, "2nd_label"] = 3
        if (
            10000 <= balanceInUSD < 100000
            and 0 < borrowInUSD / balanceInUSD < 0.2
            and numberOfLiquidation == 0
        ):
            df.loc[index, "1st_label"] = 3
            df.loc[index, "2nd_label"] = 4
        # Label 4
        if (
            balanceInUSD > 1000000
            and numberOfLiquidation == 0
            and borrowInUSD / balanceInUSD < 0.2
        ):
            df.loc[index, "1st_label"] = 4
            df.loc[index, "2nd_label"] = 4
        if balanceInUSD >= 100000 and borrowInUSD == 0 and numberOfLiquidation == 0:
            df.loc[index, "1st_label"] = 4
            df.loc[index, "2nd_label"] = 3
    df.to_csv(file_path)


add_label(df)

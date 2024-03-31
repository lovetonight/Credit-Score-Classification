import json
import pymongo
from config import MongoDBConfig
import pandas as pd

# config

config = MongoDBConfig()

# mongoDB

list_hosts = config.HOST.split(',')
hosts_port = ""
for host in list_hosts:
    hosts_port += f"{host}:{config.PORT},"
hosts_port = hosts_port.rstrip(',')
mongo_uri = f"mongodb://{config.USERNAME}:{config.PASSWORD}@{hosts_port}/"

    # connect
client = pymongo.MongoClient(mongo_uri)
db = client["knowledge_graph"]
collection = db["wallets"]

# depositors
df = pd.read_csv('full_data.csv')
tmp_df = df[["address", "label"]]
depositers, labels = df["address"], df["label"]
depositers = ["0x1_" + depositer for depositer in depositers]

# filter
# end_words = [
#     "Logs",
#     "tokens",
#     "Tokens",
#     "elite",
#     "selective",
#     "writingLock",
#     "lendings",
#     "flagged",
#     "isNew",
#     "At",
# ]
# regex_pattern = "|".join(map(re.escape, end_words))
# regex = re.compile(f".*(?:{regex_pattern})$")

projection = {
    "address": 1,
    "balanceInUSD": 1,
    "borrowInUSD": 1,
    "depositInUSD": 1,
    "createdAt": 1,
    "numberOfLiquidation": 1,
    "totalValueOfLiquidation": 1,
    "frequencyOfDappTransactions": 1,
    "numberOfInteractedDapps": 1,
    "numberOfReputableDapps": 1,
    "typesOfInteractedDapps": 1,
}

filter = {"_id": {"$in": depositers}}
cursor = collection.find(filter, projection)
cursor = list(cursor)
new_df = pd.DataFrame(cursor)

# Merge & write to csv file
df_final = pd.merge(new_df, tmp_df, on='address')
csv_file_path = "new_data.csv"
df_final.to_csv(csv_file_path, index=False)


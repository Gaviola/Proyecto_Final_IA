import numpy as np
import pandas as pd
import pickle
import json
import re
import time
#from pandas.io.json import json_normalize
from tqdm.notebook import tqdm

tqdm.pandas()

# Load the newest version of match_data pickle file
import os

file_path = 'D:\\Facultad\\3_año\\Inteligencia_Artificial_1\\Dataset'
if os.path.exists(file_path) and os.access(file_path, os.R_OK):
    print("El archivo existe y es legible")
else:
    print("No puedo acceder al archivo")

match_data_df = pd.read_pickle(file_path + '\\match_data_version2.pickle')
pd.set_option('display.max_columns', None)
print(match_data_df.head())


""" # First, we need to drop gameCreation, gameType, gameVersion, mapId, platformId, queueId, seasonId
# status.message, status.status_code
# since these are either unique value column or null or irrelevant to our app.
# It will help us to reduce the size of the file and be able to process the data furether locally
match_data_df.drop(['gameCreation', 'gameType', 'gameVersion', 'mapId',
                    'platformId', 'queueId', 'seasonId', 'status.message', 'status.status_code'], axis=1, inplace=True)

# Second, we need to remove the games data that are not CLASSIC gameMode
# since we are focusing of classic mode game (90% of the data)
indexList = [i for i in match_data_df[match_data_df.gameMode != 'CLASSIC'].index if i != 0]

print('Dropping non-classic games...')
match_data_df.drop(indexList, inplace=True)
match_data_df.drop(['gameMode'], axis=1, inplace=True)

print("Succeed!") """
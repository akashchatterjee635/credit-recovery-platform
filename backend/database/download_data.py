import pandas as pd
from sklearn.datasets import fetch_openml
import os

print('Fetching Home Credit Default Risk dataset from OpenML...')
try:
    # OpenML ID 41144 or 42721 or 45041 might be home credit. Let's search by name
    # We can fetch by name: 'Home-Credit-Default-Risk'
    dataset = fetch_openml(name='Home-Credit-Default-Risk', version=1, as_frame=True, parser='auto')
    df = dataset.frame
    df.to_csv('data/application_train.csv', index=False)
    print(f'Successfully downloaded {df.shape[0]} rows and {df.shape[1]} columns.')
except Exception as e:
    print('Failed to fetch from OpenML:', e)
    print('Please download the dataset from https://www.kaggle.com/c/home-credit-default-risk/data')
    print('and place application_train.csv in the data/ directory.')

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset", path="spam.csv")

df = pd.read_csv(path, encoding="latin-1")

xtrain, xtest, ytrain, ytest = train_test_split(df["v2"], df["v1"], test_size=0.2, random_state=42)

uq = set(xtrain.str.lower().str.findall(r"\b\w+\b").explode().dropna())

val, cnt = np.unique(xtrain.str.lower().str.findall(r"\b\w+\b").explode().dropna(), return_counts=True)

vals, cnts = np.unique(xtrain.loc[ytrain=="spam"].str.lower().str.findall("\b\w+\b").explode().dropna(), return_counts=True)



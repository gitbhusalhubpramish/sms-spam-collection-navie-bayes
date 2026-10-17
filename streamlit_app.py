import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import streamlit as st

path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset", path="spam.csv")

df = pd.read_csv(path, encoding="latin-1")

xtrain, xtest, ytrain, ytest = train_test_split(df["v2"], df["v1"], test_size=0.2, random_state=42)

uq = set(xtrain.str.lower().str.findall(r"\b\w+\b").explode().dropna())

val, cnt = np.unique(xtrain.str.lower().str.findall(r"\b\w+\b").explode().dropna(), return_counts=True)

vals, cnts = np.unique(xtrain.loc[ytrain=="spam"].str.lower().str.findall(r"\b\w+\b").explode().dropna(), return_counts=True)

pxs = pd.DataFrame({"val": vals, "cnt": cnts})
pxs["cnt"] = pxs["cnt"]/sum(pxs["cnt"])

ps = sum(cnts)/sum(cnt)

px = pd.DataFrame({"val": val, "cnt": cnt})
px["cnt"] = px["cnt"]/sum(cnt)

valh, cnth= np.unique(xtrain.loc[ytrain=="ham"].str.lower().str.findall(r"\b\w+\b").explode().dropna(), return_counts=True)

ph = sum(cnth)/sum(cnt)

pxh = pd.DataFrame({"val": valh, "cnt":cnth})
pxh["cnt"] = pxh["cnt"]/sum(pxh["cnt"])

st.write('''## Data(set)''')
st.write('''This project uses [sms spam collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) dataset in kaggle. The dataset has some sms messages labeled as spam and ham. The information of the dataset(numbers only) is:''')
st.write('''- Total words:''', sum(cnt))
st.write('''- Spam words:''', sum(cnts))
st.write('''- ham words:''', sum(cnth))

st.write('''## Algorithm''')
st.write('''This project uses naive bayes algorithm to predict whether a message is spam or ham from the dataset given. The parameter for the model are just the probability of being ham, spam and likehood. Since dataset is too large - 8000 total unique so we can't display likehood so we can display:''')
st.write('''- P(s)''', ps)
st.write('''- P(h)''', ph)




psx = xtest.apply(lambda row: (np.log(ps)+ sum(np.log(max(pxs.set_index("val")["cnt"].get(x,0), np.finfo(float).tiny)) for x in row.split())))

phx = xtest.apply(lambda row: (np.log(ph) + sum(np.log(max(pxh.set_index("val")["cnt"].get(x,0), np.finfo(float).tiny)) for x in row.split())))

pred = psx>phx
ypred = pd.Series(np.where(pred, "spam", "ham"), index=pred.index)

st.write('''## Accuracy''')
st.write(accuracy_score(ytest, ypred))

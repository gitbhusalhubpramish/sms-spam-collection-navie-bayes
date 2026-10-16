import kagglehub
import pandas as pd

path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset", path="spam.csv")

df = pd.read_csv(path)

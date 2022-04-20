import pickle

from datasets import Dataset
from lightgbm import LGBMClassifier
import pandas as pd

print("Loading Dataset....")
train_dataset = Dataset.from_file("data/trainv3.arrow")
dtypes = {}
for k, v in train_dataset.features.items():
    dtype = v.dtype
    if dtype == "bool":
        dtypes[k] = pd.CategoricalDtype([True, False])

train_df = train_dataset.to_pandas()
train_X = train_df.iloc[:, :-1].astype(dtypes)
train_y = train_df.iloc[:, -1]

valid_dataset = Dataset.from_file("data/validv3.arrow")
valid_df = valid_dataset.to_pandas()
valid_X = valid_df.iloc[:, :-1].astype(dtypes)
valid_y = valid_df.iloc[:, -1]

model = LGBMClassifier(objective="binary")

print("Train Model ....")
model.fit(train_X, train_y)

train_socre = model.score(train_X, train_y)
valid_score = model.score(valid_X, valid_y)

print(f"{train_socre=}")
print(f"{valid_score=}")

with open("model2/lgbm.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model2/dtypes.pkl", "wb") as f:
    pickle.dump(train_X.dtypes, f)
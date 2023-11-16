from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

def load_data():
    path = Path("datasets/housing.tgz")
    if not path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, path)
        with tarfile.open(path) as housing:
            housing.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

def shuffle_and_split_data(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing = load_data()

# print(housing.head())
# housing.info()
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())

# housing.hist(bins=50, figsize=(12, 8))
# plt.show()

# train_set, test_set = shuffle_and_split_data(housing, 0.2)

# print(train_set.tail(1))

# housing_with_id = housing.reset_index()
# housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1,2,3,4,5])

# housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
# plt.xlabel("Income category")
# plt.ylabel("Number of districts")
# plt.show()


# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

# strat_train_set, strat_test_set = strat_splits[0]

strat_train_set, strat_test_set = train_test_split(housing, 
                                                   test_size=0.2, 
                                                   stratify=housing["income_cat"], 
                                                   random_state=42)

# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

# print(housing.tail())

# housing.plot(kind="scatter",
#              x="longitude",
#              y="latitude",
#              grid=True,
#              s=housing["population"] / 100,
#              label="Population",
#              c="median_house_value",
#              cmap="jet",
#              colorbar=True,
#              legend=True,
#              sharex=False,
#              figsize=(10, 7))
# plt.show()

# corr_matrix = housing.corr(numeric_only=True)
# print(corr_matrix["median_house_value"].sort_values(ascending=True))

# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)
plt.show()
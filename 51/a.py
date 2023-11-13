from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_data():
    path = Path("datasets/housing.tgz")
    if not path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, path)
        with tarfile.open(path) as housing:
            housing.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_data()
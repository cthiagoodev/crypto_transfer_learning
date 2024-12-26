import pandas as pd
import tensorflow as tf
import keras as kr
from crypto_dataset import CryptoDataset


def main():
    dataframe: pd.DataFrame = get_dataframe()
    kr.models.load_model()

def get_dataframe() -> pd.DataFrame:
    dataset = CryptoDataset("https://www.kaggle.com/api/v1/datasets/download/gorgia/criptocurrencies")
    dataset.download()

    dataframe_list: list[pd.DataFrame] = []

    for ds_item in dataset.files_path:
        dataframe_list.append(pd.read_csv(ds_item))

    return pd.concat([*dataframe_list])

if __name__ == "__main__":
    main()

import argparse
import configparser
import os
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression

from logger import Logger


def main(data_path, output_model_path):
    housing_prepared = pd.read_csv(data_path + "\\housing_train_processed.csv")
    housing_labels = pd.read_csv(data_path + "\\housinglabel_train_processed.csv")
    lg = Logger(
        "../logs/train.log",
        "file reaad successfully from {}".format(data_path),
        "w",
    )
    lg.logging()
    lin_reg = LinearRegression()
    lin_reg_model = lin_reg.fit(housing_prepared, housing_labels)

    os.makedirs(output_model_path, exist_ok=True)
    pickle.dump(lin_reg_model, open(output_model_path + "\\lin_reg.pkl", "wb"))
    lg = Logger(
        "../logs/train.log",
        "lin_reg.pkl model fit successfully and stored to {}".format(output_model_path),
        "a",
    )
    lg.logging()
    print("model stored!")
    print("check logs @ ", lg.filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_path",
        type=str,
        help="description of arg1",
        nargs="?",
        default="",
    )
    parser.add_argument(
        "stored_model_path",
        type=str,
        help="description of arg2",
        nargs="?",
        default="",
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read("../setup.cfg")

    arg1 = args.train_data_path or config["DEFAULT"]["train_data_path"]
    arg2 = args.stored_model_path or config["DEFAULT"]["stored_model_path"]
    main(arg1, arg2)

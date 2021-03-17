import pandas as pd
from timeit import default_timer as timer

def read_data():
    start = timer()
    train = pd.read_csv("app/data/training.tsv", sep = "\t", encoding = "utf-8")
    test = pd.read_csv("app/data/testing.tsv", sep = "\t", encoding = "utf-8")
    end = timer()
    print ("Number of data points - train: {}, test: {}".format(train.shape[0], test.shape[0]))
    print ("Time to read data: {} seconds".format(end - start))
    return train, test


if __name__ == "__main__":
    train, test = read_data()
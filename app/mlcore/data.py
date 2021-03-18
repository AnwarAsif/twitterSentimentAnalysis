import pandas as pd
from timeit import default_timer as timer
import re, string
import nltk
# nltk.download('punkt')

def read_data():
    start = timer()
    train = pd.read_csv("app/data/training.tsv", sep = "\t", encoding = "utf-8")
    test = pd.read_csv("app/data/testing.tsv", sep = "\t", encoding = "utf-8")
    end = timer()
    print ("Number of data points - train: {}, test: {}".format(train.shape[0], test.shape[0]))
    print ("Time to read data: {} seconds".format(end - start))
    return train, test

def preprocess_data(df):
    """
    Preprocess data:
    -- strip spaces and lowercase
    -- get rid of numbers
    -- get rid of punctuations
    -- get rid of single letters

    Keyword arguments:
    df -- Input two column dataframe with 'sentence' and 'label'

    Returns:
    df -- With preprocessed 'sentence' column and 'label'
    """

    def preprocessor(sentence):
        sentence = sentence.strip().lower()
        sentence = re.sub(r"\d+", "", sentence)
        sentence = sentence.translate(sentence.maketrans(string.punctuation, ' '*len(string.punctuation)))
        sentence = " ".join([w for w in nltk.word_tokenize(sentence) if len(w) > 1])
        
        return sentence

    start = timer()
    df["sentence"] = df["sentence"].apply(preprocessor)
    end = timer()
    print ("Time to preprocess {} data points: {} seconds".format(df.shape[0], end - start))
    return df


if __name__ == "__main__":
    train, test = read_data()
    train = preprocess_data(train)
    test = preprocess_data(test)
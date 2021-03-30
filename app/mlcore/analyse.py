import os
from joblib import load
import pandas as pd 
import numpy as np

clf = load('app/mlcore/grid.joblib')
vec = load('app/mlcore/vec.joblib')

def sentiment_analyzer(tweet):

    sample_tweet = pd.Series(tweet)
    sample_tweet = vec.transform(sample_tweet)
    result = clf.predict(sample_tweet)

    return str(result[0])

if __name__ == "__main__":
    sample_tweet = "In December alone , the members of the Lithuanian Brewers ' Association sold a total of 20.3 million liters of beer , an increase of 1.9 percent"
    result = sentiment_analyzer(sample_tweet)
    print(result)
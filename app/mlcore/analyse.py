def sentiment_analyzer(tweet):
    if len(tweet) >=20: 
        return 'positive'
    if len(tweet) <20 and len(tweet) >10:
        return 'negative'
    else:
        return 'neutral'

if __name__ == "__main__":
    sample_tweet = "In December alone , the members of the Lithuanian Brewers ' Association sold a total of 20.3 million liters of beer , an increase of 1.9 percent"
    result = sentiment_analyzer(sample_tweet)
    print(result)
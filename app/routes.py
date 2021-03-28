import os, sys
from app import app 
from flask import render_template, jsonify, request
from .mlcore import *

@app.route('/',methods=['GET','POST'])
def index():
    tweet = request.args.get('tweet')
    if tweet:
        result = sentiment_analyzer(tweet)
        return render_template('index.html', result=result, tweet=tweet)
    else: 
        return render_template('index.html')

@app.route('/api')
def index_with_name():
    return render_template('api.html')

@app.route('/analyze/<string:tweet>')
def tweet_analyzer(tweet):
    result = sentiment_analyzer(tweet)
    return jsonify(result=result)

if __name__ == "__main__":
    sample_tweet = "In December alone , the members of the Lithuanian Brewers ' Association sold a total of 20.3 million liters of beer , an increase of 1.9 percent"
    result = sentiment_analyzer(sample_tweet)
    print(result)

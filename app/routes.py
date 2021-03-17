import os, sys
from app import app 
from flask import render_template, jsonify
from .mlcore import *

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<name>')
def index_with_name(name):
    return render_template('index.html',name=name)

@app.route('/analyze/<string:tweet>')
def tweet_analyzer(tweet):
    result = sentiment_analyzer(tweet)
    return jsonify(result=result)

if __name__ == "__main__":
    sample_tweet = "In December alone , the members of the Lithuanian Brewers ' Association sold a total of 20.3 million liters of beer , an increase of 1.9 percent"
    result = sentiment_analyzer(sample_tweet)
    print(result)

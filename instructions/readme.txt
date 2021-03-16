Welcome to Atradius' Data Scientist Interview Process!


Take-home task:

One of the many different data science tasks, projects and proof-of-concepts we do at Atradius, is that of measuring the general sentiment levels towards companies Atradius has exposure to. This helps Atradius stay on top of future risks, by monitoring whether a company is suddenly getting a lot of negative reviews on, for example, social media.
For your take-home task, we reduce this problem to the simple task of sentence-based sentiment classification. This means that given a sentence, a classifier must say whether the sentiment conveyed in it is "positive", "negative" or "neutral".

The dataset that has been provided to you has two parts:
- training.tsv
- testing.tsv
which have an identical two-column tab-separated format: both of them they have "sentence"s and "label"s.
You should not use the testing data for any training purposes.


Task 1 (Python):
Train your own classifier
- Train a machine learning classifier that can classify sentences into one of three types of sentiments, as mentioned above
- To get you started, we have provided a baseline logistic regression classifier (baseline.py). Ideally, your classifier should be able to get better performance than this rudimentary baseline



Task 2 (Python):
Build a classification API
- Once you train a classifier, we want you to make a REST-API with the trained classifier
- We recommend using Flask to make your API
- The API should ingest a POST request with a sentence and return a classification label, i.e., "positive", "negative" or "neutral".


You must return all your code which can:
- train a classifier
- deploy a trained classifier locally
- have an example script to call your API with some example sentences and get their labels from the classifier
We will run your code, so make sure it can be run without too many changes.

Deadline for finishing this is April 1st 2021.

If your solution qualifies, on the day of the interview we would like you to present your solution and code to us, and rationalize the choices you have made in order to achieve your results.

We aren't looking for a perfect solution. People can spend months creating the best sentiment analysis. Please do not do this. We want to see your enthusiasm, rationale and coding quality. So please don't spend more than 2 days on this. And try to have fun too!
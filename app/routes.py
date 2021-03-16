from app import app 
from flask import render_template

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<name>')
def index_with_name(name):
    return render_template('index.html',name=name)
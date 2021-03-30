# Tweet Analyser 
A simple Application to classify sentiment of tweets. 

## The main Structure of the Application 

### Environment
The application is build using Flask. Please install all required packages in your environment to run the application. The application can be launched using  running below command.
`python run.py`.  

- A live implementation can be found at below url: 
https://flasktwitter.herokuapp.com/ 
- Full source code can be found at: https://github.com/AnwarAsif/twitterSentimentAnalysis

### Part 1: MLCore 
The core components of machine learning algorithms
- **Training:** The training is done by using the help of three python files. 
    - `training.py`: the main training file. 
    - `data.py`: load data and preprocessing 
    - `models.py`: base line model, best model and grid search function to search model. 
- **Analyse:** `analyse.py` python file is created give frontend to have access to the build model. 

### Part 2: Restful API 
A simple restful API is build to provide access to the selected ML model. 
- `URL Access`: https://flasktwitter.herokuapp.com/analyse/
- `API Documentation`: https://flasktwitter.herokuapp.com/api 



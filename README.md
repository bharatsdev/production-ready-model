[![Build Status](https://dev.azure.com/bharatmca2010/bharatmca2010/_apis/build/status/everythingisdata.Productionize-Machine-Learning?branchName=master)](https://dev.azure.com/bharatmca2010/bharatmca2010/_build/latest?definitionId=3&branchName=master)

[![Build Status](https://travis-ci.com/everythingisdata/Productionize-Machine-Learning.svg?branch=master)](https://travis-ci.com/everythingisdata/Productionize-Machine-Learning)
# Productionize Machine Learning Project
This will help us in productionized of the Machine Learning model. 
Also
- ML Model Design
- Pickle of ML model (Serialized)
- Flask Rest Endpoint
- Containerized the ML Model with Docker
- Deploy on AWS Beanstalk
 
## Technology Stack
   - Python 3
   - Scikit-Learn
   - Numpy
   - Pandas
   - Flask
   - AWS
   - Docker
## Getting Start with the Machine Learning as a service
  - Data Preparation & Cleaning
  - Creating model
  - Analysis of data
  - Correlation
  - missing values
  - Drive Features
### Data Preparation & Cleaning
- Correction
- Correlation
- Converting
- Fixing
- Classifying  
### Create a Model
- Use the Scikit-learn model for training Model and fit and train the model. 
   <!-- - Serialized the model
   - Create Flask Rest Endpoint
   - Load Serialized model in Flask app
   - pass the JSON object to Flask request
   - Deploy on AWS as Service --> 
### Design a ML Pipeline
- Pipeline make the chains of several step together.
- Scikit learn have a good class Pipeline this usesfor Creating the Pipeline for data pre-processing and joining all the pipe in serial form. 
- from sklearn.pipeline import make_pipeline .


### requirements.txt file
- Requirements files" are files containing a list of items to be installed using pip install like so
- pip install -r requirements.txt
- pip freeze > requirements.txt
### Docker Configuration  
- docker build .
- docker run image <IMGAGE>
- docker build -t ml-as-a-service:latest .
- docker run -d -p 5000:5000 ml-as-a-service
- docker ps  : check running images

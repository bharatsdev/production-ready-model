# Machine Learning As A Service
   - Create a Model
   - Design a Pipeline
   - Serialized the model
   - Create Flask Rest Endpoint
   - Load Serialized model in  Flask app
   - pass the JSON object to Flask request
   - Deploy on AWS  as Service

## Technologies Stack
   - Python 3
   - Scikit-Learn
   - Numpy
   - Pandas
   - Flask
   - AWS
   - Docker
### Getting Start with the Machine Learning as a service
  - Creating model
  - Analysis of data
  - correlation
  - missing values
  - Drive Features
  ##### Design a Pipeline   
        - sklearn have a good class Pipeline this user for Creating the Pipeline for data pre-processing and joining all the pipe in serial form.
        from sklearn.pipeline import make_pipeline .
  ##### Create requirements.txt file
    Requirements files" are files containing a list of items to be installed using pip install like so
    - pip install -r requirements.txt
    - pip freeze > requeriments.txt
  ##### Created Dockerfile
     - docker build .
     - docker run image <IMGAGE>
     - docker build -t ml-as-a-service:latest .
     - docker run -d -p 5000:5000 ml-as-a-service
     - docekr ps  : check running images
     - 

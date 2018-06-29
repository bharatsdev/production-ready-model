FROM python:3
MAINTAINER Bharat Singh "bharatmca2010@gmail.com" 
# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /aiwizards/requirements.txt
WORKDIR /aiwizards
RUN pip install -r requirements.txt
COPY . /aiwizards
CMD [ "python", "./aitechwizardsMaaS.py" ]



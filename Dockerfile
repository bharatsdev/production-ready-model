FROM python:3.6-slim

MAINTAINER Bharat Singh "bharatmca2010@gmail.com"

# Update Packages
RUN apt-get update -y

# Install Python setuptools
RUN apt-get install -y python-setuptools

# Install pip
RUN easy_install  pip

# Copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /aiwizards/requirements.txt

WORKDIR /aiwizards

# Install Python modules
RUN pip install -r requirements.txt

#Bundle app source
COPY . /aiwizards

# tell the port number the container should expose
EXPOSE 5000

CMD [ "python", "./aitechwizardsMaaS.py" ]

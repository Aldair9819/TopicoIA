# Use the latest Python Version as the base image
FROM tensorflow/tensorflow:2.11.0rc1-gpu-jupyter
#FROM tensorflow/tensorflow:latest-gpu-jupyter
WORKDIR /tf-knugs
RUN apt-get update && \
apt-get upgrade -y

# Copy the requirements file to the container
COPY ./requirements.txt ./

RUN apt-get install -y --fix-missing \
    cmake \
    git \
    wget \
    ffmpeg 

# Install the Python dependencies using Python 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
EXPOSE 8889

ENTRYPOINT [ "jupyter","notebook","--ip=0.0.0.0","--port=8889","--allow-root", "--no-browser" ]

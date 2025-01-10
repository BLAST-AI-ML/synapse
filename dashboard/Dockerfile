# Use an official Python runtime as a parent image
FROM continuumio/miniconda3:latest

# NOTE Uncomment this block to enable CUDA
# Use an official CUDA-enabled base image
#FROM nvidia/cuda:12.6.3-base-ubuntu24.04
# Install Miniconda
#RUN apt-get update && \
#    apt-get install wget -y && \
#    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#    bash /tmp/miniconda.sh -b -p /opt/conda && \
#    rm /tmp/miniconda.sh
# Set the PATH to include Miniconda
#ENV PATH=/opt/conda/bin:$PATH

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in environment.yml
RUN conda env create -f gui.yml

# TODO Remove if not persistent through container launch
# Activate the environment
#SHELL ["conda", "run", "-n", "gui", "/bin/bash", "-c"]

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Set environment variable to run GUI as desktop application
ENV PYWEBVIEW_GUI=qt

# NOTE Need to activate conda environment at container launch
# Run app.py when the container launches
CMD ["conda", "run", "-n", "gui", "python", "app.py", "--app"]

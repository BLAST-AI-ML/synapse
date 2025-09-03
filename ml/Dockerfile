# Use an official Python runtime as a parent image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app/ml

# Silence pip error to install in root dirs
ENV PIP_ROOT_USER_ACTION=ignore

# Install any needed packages specified in the environment file
COPY ml/environment.yml /app/ml/environment.yml
RUN conda env create -y -f environment.yml \
    && conda clean --all -y

# Configure an exectuable entrypoint script
COPY ml/entrypoint.sh /app/ml/entrypoint.sh
RUN chmod +x /app/ml/entrypoint.sh

# Define the entrypoint to activate the environment in interactive and CMD use
ENTRYPOINT ["/app/ml/entrypoint.sh"]

# Copy ML scripts & configs into the container at /app/ml/
COPY ml/environment.yml /app/ml/environment.yml
COPY ml/train_model.py /app/ml/train_model.py
COPY ml/Neural_Net_Classes.py /app/ml/Neural_Net_Classes.py
COPY dashboard/config/variables.yml /app/ml/variables.yml

# Run train_model.py when the container launches
CMD ["python", "-u", "train_model.py"]

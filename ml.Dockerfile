# Use an official Python runtime as a parent image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app/ml

# Silence pip error to install in root dirs
ENV PIP_ROOT_USER_ACTION=ignore

# Install any needed packages specified in the environment file
# Match the CUDA 12.4.0 on Perlmutter (NERSC):
#   https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#specifying-a-different-target-platform-for-an-environment
COPY ml/environment.yml /app/ml/environment.yml
ENV CONDA_OVERRIDE_CUDA=12.4.0
RUN conda info
RUN conda env create -y -f environment.yml \
    && conda clean --all -y

# Configure an exectuable entrypoint script
COPY ml/entrypoint.sh /app/ml/entrypoint.sh
RUN chmod +x /app/ml/entrypoint.sh

# Define the entrypoint to activate the environment in interactive and CMD use
ENTRYPOINT ["/app/ml/entrypoint.sh"]

# Copy ML scripts & configs into the container at /app/ml/
COPY ml/train_model.py /app/ml/train_model.py
COPY ml/Neural_Net_Classes.py /app/ml/Neural_Net_Classes.py
COPY experiments /app/experiments

# Run train_model.py when the container launches
CMD ["python", "-u", "train_model.py"]

# Use an official Python runtime as a parent image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app/ml

# Copy content into the container at /app
COPY ml/environment.yml /app/ml/environment.yml
COPY ml/train_model.py /app/ml/train_model.py
COPY ml/Neural_Net_Classes.py /app/ml/Neural_Net_Classes.py
COPY ml/training_pm.sbatch /app/ml/training_pm.sbatch
COPY dashboard/config/variables.yml /app/ml/variables.yml

# Install any needed packages specified in the environment file
RUN conda env create -y -f environment.yml \
    && conda clean --all -y

# Copy the entrypoint script
COPY ml/entrypoint.sh /app/ml/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /app/ml/entrypoint.sh

# Define the entrypoint to activate the environment in interactive and CMD use
ENTRYPOINT ["/app/ml/entrypoint.sh"]

# Run train_model.py when the container launches
CMD ["python", "-u", "train_model.py"]

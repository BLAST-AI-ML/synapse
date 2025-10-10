# syntax=docker/dockerfile:1.7-labs
# Declare the syntax version to use for the build,
# as the --exclude flag in the COPY command is not
# yet available in stable syntax:
# - https://docs.docker.com/reference/dockerfile/#syntax
# - https://docs.docker.com/reference/dockerfile/#copy---exclude

# Use an official Python runtime as a parent image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app/dashboard

# Silence pip error to install in root dirs
ENV PIP_ROOT_USER_ACTION=ignore

# Install any needed packages specified in the environment file
COPY dashboard/gui-lock.yml /app/dashboard/gui-lock.yml
RUN conda install -c conda-forge conda-lock \
    && conda-lock install --name gui gui-lock.yml \
    && conda clean --all -y

# Configure an exectuable entrypoint script
COPY dashboard/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Define the entrypoint to activate the environment in interactive and CMD use
ENTRYPOINT ["/entrypoint.sh"]

# Copy content into the container at /app
COPY --exclude=*.pem dashboard /app/dashboard
COPY ml/train_model.py /app/ml/train_model.py
COPY ml/Neural_Net_Classes.py /app/ml/Neural_Net_Classes.py
COPY ml/training_pm.sbatch /app/ml/training_pm.sbatch

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "-u", "app.py", "--timeout", "0", "--port", "8080", "--host", "0.0.0.0", "--server"]
# The URL http://0.0.0.0:8080/ is used by the application to indicate that
# it is listening on all network interfaces inside the container, but you
# should access it via http://localhost:8080/ from your host machine)

# Use an official Python runtime as a parent image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in the environment file
RUN conda install -c conda-forge conda-lock \
    && conda-lock install --name gui gui-lock.yml

# Activate env in interactive and CMD use
COPY entrypoint.sh /entrypoint.sh
RUN echo "source entrypoint.sh" >> ~/.bashrc
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
# NOTE The URL http://0.0.0.0:8080/ is used by the application to indicate that
#      it is listening on all network interfaces inside the container, but you
#      should access it via http://localhost:8080/ from your host machine
CMD ["python", "app.py", "--port", "8080", "--host", "0.0.0.0", "--server", \
     "--model", "/app/ml/NN_training/base_simulation_model_with_transformers_calibration.yml"]

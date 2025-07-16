# Use a lean, official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /code

# --- ADD THIS LINE ---
# Install the missing system dependency for GDAL/rasterio
RUN apt-get update && apt-get install -y libexpat1

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application's code to the working directory
COPY ./app /code/app

# Command to run the application using Uvicorn
# The host must be 0.0.0.0 to be reachable from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
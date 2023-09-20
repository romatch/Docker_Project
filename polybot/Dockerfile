# Use an official Python runtime as a parent image
FROM python:3.10-alpine

# Set the working directory to /app
WORKDIR /app

# Copy the requirements
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Make port 8443 available to the world outside this container
EXPOSE 8443

# Run app.py when the container launches
CMD ["python3", "app.py"]

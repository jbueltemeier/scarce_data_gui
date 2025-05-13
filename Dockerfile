# Use an official Python image as the base image
FROM python:3.12-slim

RUN apt-get update && apt-get install -y graphviz
RUN pip install graphviz

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . /app/

# Expose the port that Streamlit will run on
EXPOSE 8080

# Set the command to run the Streamlit app
CMD ["python", "-m", "streamlit", "run", "streamlit_pages/Home.py", "--server.port=8080", "--server.address=0.0.0.0"]
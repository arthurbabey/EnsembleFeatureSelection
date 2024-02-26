# Base image
FROM python:3.11

# set the working directory in the container
WORKDIR /app

# Copy the requirement file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy all files to the WD
COPY . .

# run the application
CMD ["python", "app.py"]

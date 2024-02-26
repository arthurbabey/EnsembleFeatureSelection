# Base image
FROM python:3.11

# install R for rpy2
RUN apt-get update && apt-get install -y r-base && Rscript -e "install.packages('stabm', repos='https://cloud.r-project.org')"

# set the working directory in the container
WORKDIR /app

# Copy the requirement file into the container
COPY requirements.txt /app

# Install dependencies
RUN pip install -r requirements.txt

# Copy all files to the WD
COPY . /app

# run the application
CMD ["python", "main.py"]

# Base image
FROM python:3.11

# set WD
WORKDIR /app

# install R for rpy2
RUN apt-get update && apt-get install -y r-base && Rscript -e "install.packages('stabm', repos='https://cloud.r-project.org')"

# install python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy all other files
COPY . .

# run the application
CMD ["python", "main.py"]

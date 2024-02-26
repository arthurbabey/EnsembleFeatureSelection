# Base image
FROM python:3.11

# set WD
WORKDIR /app

# install R for rpy2
RUN apt-get update && apt-get install -y r-base && Rscript -e "install.packages('stabm', repos='https://cloud.r-project.org')"

# install python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy specific files
COPY main.py /app/
COPY utils.py /app/

# Copy directories
COPY src /app/src/
COPY tests /app/tests/

# run the application
CMD ["python", "-u", "main.py"]

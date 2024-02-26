# Base image
FROM python:3.11 AS builder

# set WD
WORKDIR /app

# install R for rpy2
RUN apt-get update && apt-get install -y r-base && Rscript -e "install.packages('stabm', repos='https://cloud.r-project.org')"

# install python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Stage 2: Production environment
FROM python:3.11:slim AS production

WORKDIR /app

# Copy all files to the WD
COPY --from=builder /app .

# run the application
CMD ["python", "main.py"]

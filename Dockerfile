FROM python:3.10-slim

#name the container
LABEL name="turbodiff"

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 0. Install system dependencies (XFOIL)
# We do this before copying Python files to take advantage of Docker layer caching.
# The 'rm -rf /var/lib/apt/lists/*' step is a Docker best practice to keep the image size slim.
RUN apt-get update && apt-get install -y \
    xfoil \
    && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt ./

# . requirements.txt only for docker image fast compile
# Docker will now cache this step permanently unless requirements.txt changes!
RUN pip install --no-cache-dir -r requirements.txt


# 1. Copy only the files needed for installation first
COPY pyproject.toml ./

COPY src ./src

ENV PYTHONPATH=/app/src

# 2. Install the production dependencies defined in [project]
RUN pip install --no-cache-dir --no-build-isolation .

# 3. Expose the port FastAPI/Uvicorn will run on
EXPOSE 8000

# 4. Run the application
# Note: Since the package is installed, 'turbodiff' is now in the site-packages
ENTRYPOINT ["uvicorn", "turbodiff.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
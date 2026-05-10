FROM python:3.11-slim

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
    libgfortran5 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# 1. Copy and install dependencies 
COPY pyproject.toml ./

# 2. Copy source code 
COPY src ./src

RUN pip install --no-cache-dir .


# 3. Expose the port FastAPI/Uvicorn will run on
EXPOSE 8000

# 4. Run the application
# Note: Since the package is installed, 'turbodiff' is now in the site-packages
ENTRYPOINT ["uvicorn", "turbodiff.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "300", "--ws-ping-timeout", "300"]
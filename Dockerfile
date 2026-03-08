FROM python:3.10-slim

#name the container
LABEL name="turbodiff"

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app



# 1. Copy only the files needed for installation first
COPY pyproject.toml ./

COPY src ./src

# 2. Install the production dependencies defined in [project]
RUN pip install --no-cache-dir .

# 3. Expose the port FastAPI/Uvicorn will run on
EXPOSE 8000

# 4. Run the application
# Note: Since the package is installed, 'turbodiff' is now in the site-packages
ENTRYPOINT ["uvicorn", "turbodiff.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
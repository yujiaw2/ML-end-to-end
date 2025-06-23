# Dockerfile for a FastAPI application

# # Step 1: Use an official lightweight Python image
# FROM python:3.11-slim

# # Step 2: Set the working directory inside the container
# WORKDIR /app

# # Step 3: Copy the entire project into the container
# COPY . /app

# # Step 4: Upgrade pip and install Python dependencies
# RUN pip install --upgrade pip \
#     && pip install --no-cache-dir -r requirements.txt

# # Step 5: Expose the port that FastAPI will run on
# EXPOSE 8000

# # Step 6: Define the default command to run the FastAPI app
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Dockerfile, layered
# Step 1: Use an official lightweight Python image
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy only the dependency file first (for cache optimization)
COPY requirements.txt /app/requirements.txt

# Step 4: Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the application code
COPY . /app

# Step 6: Expose the port that FastAPI will run on
EXPOSE 8000

# Step 7: Define the default command to run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Step 1: Use an official Python runtime as the base image
FROM python:3.10-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the requirements file to install dependencies
COPY requirements.txt .

# Step 4: Install the dependencies (including FAISS, LlamaIndex, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the application code to the working directory
COPY . /app

# Step 7: Define the entrypoint to run the application (adjust this depending on your app)
CMD ["python", "app.py"]
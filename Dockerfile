# Step 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . /app

# Step 4: Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose the port the app runs on (Flask default is 5000)
EXPOSE 5000

# Step 6: Define environment variable to indicate production mode (optional)
ENV FLASK_ENV=production

# Step 7: Run the Flask app when the container starts
CMD python ./m_tech_bits_mlops/src/app.py

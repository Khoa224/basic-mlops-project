#official Python 3.9 image
FROM python:3.9

#set the working directory
WORKDIR /app

#add app.py and models directory
COPY app.py .
COPY models/ ./models/

# add requirements file
COPY requirements.txt .

# install python libraries
RUN pip install --no-cache-dir -r requirements.txt

# specify default commands
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
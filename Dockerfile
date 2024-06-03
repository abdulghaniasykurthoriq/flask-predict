# Gunakan image Python sebagai base image
FROM python:3.9-slim

# Set direktori kerja dalam container
WORKDIR /app

# Copy requirements.txt dan install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy semua file dari direktori kerja lokal ke dalam container
COPY . .

# Ekspos port yang digunakan oleh Flask
EXPOSE 5000

# Tentukan command untuk menjalankan aplikasi
CMD ["python", "app.py"]

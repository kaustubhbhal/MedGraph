FROM python:3.9-slim

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy model and prediction script
COPY trainModel.py /app/trainModel.py
COPY predict.py /app/predict.py
COPY md_data/conditions.csv /app/md_data/conditions.csv
COPY best_model.pt /app/best_model.pt
COPY patient_graphs.pt /app/patient_graphs.pt

WORKDIR /app
CMD ["python", "predict.py"]

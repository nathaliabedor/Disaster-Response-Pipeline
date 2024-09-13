# Disaster Response Pipeline Project

## Summary

This project is part of the Data Scientist Nanodegree. The goal is to build a machine learning pipeline that categorizes real-world disaster messages. These messages are sent during natural disasters, and the task is to automatically classify them into multiple categories so they can be directed to the appropriate relief agency.

The pipeline:
- Extracts data from a SQLite database.
- Cleans and processes the data.
- Trains a machine learning model using a grid search pipeline.
- Evaluates the model's performance.
- Deploys a web app to classify new messages in real-time.

## File Descriptions

- **data/**: This folder contains the ETL pipeline.
  - `process_data.py`: Script to load, clean, and save the data into a SQLite database.
  - `DisasterResponse.db`: The SQLite database containing the cleaned data.
  - `messages.csv`, `categories.csv`: Datasets containing disaster-related messages and their corresponding categories.

- **models/**: This folder contains the ML pipeline.
  - `train_classifier.py`: Script to train a machine learning model and save it as a pickle file.
  - `classifier.pkl`: The trained machine learning model saved as a pickle file.

- **app/**: This folder contains the web application.
  - `run.py`: Flask app to run the web application for classifying new disaster messages.
  - `templates/`: Contains the HTML template files (`master.html` and `go.html`) for the web app interface.

- **notebooks/**: Jupyter notebooks used for experimentation and testing.
  - `ETL Pipeline Preparation.ipynb`: Notebook used to build and test the ETL pipeline.
  - `ML Pipeline Preparation.ipynb`: Notebook used to build and test the machine learning pipeline.

- **README.md**: This file, providing an overview of the project, how to use the scripts, and explanations of the files in the repository.

## How to Run

### 1. Running the ETL Pipeline

```bash
cd data
python process_data.py messages.csv categories.csv DisasterResponse.db

cd models
python train_classifier.py ../data/DisasterResponse.db classifier.pkl

cd app
python run.py

pip install -r requirements.txt






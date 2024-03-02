# Wearable Device Health Prediction Model

## Overview

In this project we developed anomaly detection model that catches the chaeges in person's body health based on data provided by wearable devices

## Getting Started

### Prerequisites
- Python 3
- Flask
- Scikit - learn
- Joblib

```bash
# Example installation command
pip install -r requirements.txt
```

### Installation

 To install and set up this project on your local machine:

```bash
git clone https://github.com/USG-APEK-Hackathon/DataScienc.git
cd DataScience
pip install -r requirements.txt
```

### Usage

To run this project

```bash
python app.py
```

## Project Structure

```plaintext
/
|-- app.py
|-- predict_anomaly.py
|-- isolation_forest_model.joblib
|-- scaler.joblib
|-- README.md
|-- requirements.txt
```

## Model Training

We thrained model based on 2300+ samples. We used the IsolationForest Anomaly detection model to identify changes in human health.

## External Libraries
- Flask
- scikit-learn
- joblib


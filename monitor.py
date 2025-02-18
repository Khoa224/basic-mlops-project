import logging
import joblib
import pandas as pd
from steps.clean import Cleaner
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently import ColumnMapping
import warnings
warnings.filterwarnings('ignore')

# Load the model
# import from model
# model = joblib.load("models/model.pkl")
# import from experiment runs or model registry
import mlflow
logged_model = 'runs:/6dca28337300487c979120b4e4f02d57/model'
# logged_model = 'models:/insurance_model/1'
mode = mlflow.pyfunc.load_model(logged_model)

# Load the data
reference = pd.read_csv('data/train.csv')
current = pd.read_csv('data/test.csv')
production = pd.read_csv('data/production.csv')

# Clean the data
clean = Cleaner()
reference = clean.clean_data(reference)
reference['prediction'] = mode.predict(reference.iloc[:, :-1])

current = clean.clean_data(current)
current['prediction'] = mode.predict(current.iloc[:, :-1])

production = clean.clean_data(production)
production['prediction'] = mode.predict(production.iloc[:, :-1])

# Column Mapping
target = 'Result'
prediction = 'prediction'
numerical_features = ['Age', 'Annual_Premium', 'HasDrivingLicense', 'RegionID', 'Switch']
categorical_features = ['Gender',  'PastAccident']
column_mapping = ColumnMapping()

column_mapping.target = target
column_mapping.prediction = prediction
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features

data_drift_report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    TargetDriftPreset()
])

data_drift_report.run(reference_data=reference, current_data=production, column_mapping=column_mapping)
data_drift_report
# data_drift_report.json()
data_drift_report.save_html("production_drift.html")
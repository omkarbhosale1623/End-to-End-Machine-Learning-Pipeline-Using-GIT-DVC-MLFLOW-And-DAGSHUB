import pandas as pd
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/omkarbhosale1623/machinelearningpipeline.git"
os.environ['MLFLOW_TRACKING_USERNAME']="omkarbhosale1623"
os.environ['MLFLOW_TRACKING_PASSWORD']="e25f3586a7079899fc99c3a0db6576ae3e4b20d2"

# Load the parameters from params.yaml
params=yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data["Outcome"]
    
    mlflow.set_tracking_uri("https://dagshub.com/omkarbhosale1623/machinelearningpipeline.mlflow")
    
    # Load the Model from the disk
    model=pickle.load(open(model_path,'rb'))
    
    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)
    
    mlflow.log_metric('accuracy',accuracy)
    print('model accuracy',accuracy)
    
if __name__=="__main__":
    evaluate(params['data'], params['model'])
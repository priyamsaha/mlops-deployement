# model.py
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model.pkl"

def train_and_save_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = RandomForestClassifier()
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    print("Model trained and saved.")

def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    return joblib.load(MODEL_PATH)

def predict(features):
    model = load_model()
    return model.predict([features]).tolist()

import joblib


def predict(data):
    clf = joblib.load("modelKNN1.pkl")
    return clf.predict(data)
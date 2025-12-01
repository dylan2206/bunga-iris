import joblib


def predict(data):
    clf = joblib.load("rd_model.sav")
    return clf.predict(data)
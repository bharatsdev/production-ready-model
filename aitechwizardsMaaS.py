import flask
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.externals import joblib
from model.ModelTrainTest import PreProcessing

# Machine Learning As A Service App
mlaasapp = Flask(__name__)


@mlaasapp.route("/")
def welcome():
    return (" This is my first Machine Learning Model As A service")


@mlaasapp.route("/api/predict/", methods=['POST'])
def predictApiCall():
    """Predication API Call
    Pandas DataFrame sent as payload in Post Request """
    print("Service Request")
    try:
        # Will Read the data from Request as Json
        requestJson = request.get_json()
        # print(requestJson)
        # Convert Json to Panda DataFrame
        test_df = pd.DataFrame(requestJson)
        # print(test_df)
        # Getting the Loan ID seprated
        PassengerName = test_df['Name']
    except Exception as e:
        raise e
    finally:
        print("Final >>")
    fileName = './data/aitechwizard.pkl'

    if test_df.empty:
        return (bad_request())
    else:
        # Load the Saved Model
        print("Loading the model...")
        clf = joblib.load(fileName)
        print("Your Model have been loading successfully ... Doing Predication now...")
        print(test_df.columns[test_df.isna().any()].tolist())
        test_df.Age = test_df.Age.astype(int)

        test_pred = clf.predict(test_df)
        test_pred_series = list(pd.Series(test_pred))
        final_prediciton = pd.DataFrame(list(zip(PassengerName, test_pred_series)))
        response = jsonify(predictions=final_prediciton.to_json(orient="records"))
        response.status_code = 200
        return (response)


@mlaasapp.errorhandler(400)
def bad_request():
    message = {"status": 400,
               "message": " Bad request" + request.url + " . Kindly check you  input data"}
    resp = jsonify(message)
    resp.Status = 400
    return resp


if __name__ == '__main__':
    mlaasapp.run(host="0.0.0.0", debug=True, port=80)

# import necessary packages 
import pandas as pd
import numpy as np
import sklearn
import joblib
from flask import Flask, render_template, request

# let the name of the app be the name of this file
app = Flask(__name__) 

# bind an endpoint or url to a function using a decorator
# when the url runs, this function is called
@app.route("/") # this is the home page
def home():
    return render_template("input.html")

@app.route("/output", methods=["GET", "POST"])
def output():
    # if this url was requested through a post method
    if request.method == "POST":
        #print(request.form["house_age"])
        # extract the data that was submitted to this url
        input_data = [float(x) for x in request.form.values()]
        # get the data ready to be used for a prediction
        input_data = np.array(input_data).reshape(1, -1)
        # load the saved model 
        model = open("lin_reg.pkl", "rb")
        model = joblib.load(model)
        # make a prediction 
        pred = model.predict(input_data)
        pred = round(float(pred), 2)
        return render_template("output.html", prediction=pred)
    return render_template("output.html")
if __name__=="__main__":
    app.run(debug=True)
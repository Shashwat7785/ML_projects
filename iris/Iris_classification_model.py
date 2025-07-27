import numpy as np
from flask import Flask , render_template,url_for, request
import pickle
app = Flask(__name__)
model = pickle.load(open("knn_model.pkl","rb"))
l = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
flower = {"Iris-setosa" : "iris_setosa.jpg" , "Iris-versicolor" : "Iris_versicolor.jpg","Iris-virginica" : "Iris_virginica.jpg" }
@app.route("/")
def home():
     return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
     features = [float(x) for x in request.form.values()]
     f = np.array([features])
     prediction = model.predict(f)
     image = flower[l[prediction[0]]]
     return render_template("index.html", prediction_text = "The flower is {}".format(l[prediction[0]]), prediction_image = image)


if __name__ == "__main__":
    app.run(debug = True)
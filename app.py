from flask import Flask,request,render_template,url_for
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route("/")
def man():
    return render_template('iris.html')



@app.route("/predict", methods = ['POST'])
def home():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    predict = model.predict(final_features)

    return render_template('base.html',data=predict)


if __name__ == "__main__":
    app.run(debug=True)


from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('Flask-API/data/modelDiscriminant.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict_placement():
    AMT_INCOME_TOTAL = float(request.form.get('AMT_INCOME_TOTAL'))
    AMT_CREDT = int(request.form.get('AMT_CREDT'))
    NAME_INCOME = int(request.form.get('NAME_INCOME'))

    # prediction
    result = model.predict(
        np.array([AMT_INCOME_TOTAL, AMT_CREDT, NAME_INCOME]).reshape(1, 3))

    result = 'placed' if result[0] == 1 else 'not placed'

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
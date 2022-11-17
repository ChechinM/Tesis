from flask import Flask, render_template, request
from waitress import serve
import pickle
import numpy as np

model = pickle.load(open('data/modelDiscriminant.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test', methods=['GET'])
def getResult():
    input = np.array([202500.00, 406597.50, 7, 4, 14, 0, 66]).reshape(1, 7)  # 0 -> para clientes no morosos | 1 -> para clientes morosos

    # prediction
    result = model.predict(input)
    print("Resultado : " + str(result))

    result = 'APLICA' if result[0] == 0 else 'NO APLICA'

    return render_template('index.html', result=result)


@app.route('/predict', methods=['POST'])
def predict_placement():
    AMT_INCOME_TOTAL = float(request.form.get('AMT_INCOME_TOTAL'))
    AMT_CREDT = float(request.form.get('AMT_CREDT'))
    NAME_INCOME = int(request.form.get('NAME_INCOME'))
    NAME_EDUCATION = int(request.form.get('NAME_EDUCATION'))
    TERM_MONTH = int(request.form.get('TERM_MONTH'))
    CNT_CHILDRE = int(request.form.get('CNT_CHILDRE'))
    AGE = int(request.form.get('AGE'))

    # prediction
    result = model.predict(np.array([AMT_INCOME_TOTAL, AMT_CREDT, NAME_INCOME, NAME_EDUCATION, TERM_MONTH, CNT_CHILDRE, AGE]).reshape(1, 7))
    result = 'APLICA' if result[0] == 0 else 'NO APLICA'

    return render_template('index.html', result=result)


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)

from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model and the encoder at startup



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_data', methods = ['POST'])
def model_prediction():
    data = request.form 
    print(data)

    model = pickle.load(open(r'logistic_model.pkl','rb'))
    print(model)

    user_data = [[float(data['age']),
                  float(data['length_of_service']),
                  float(data['avg_training_score']),
                  float(data['awords_won'])
                  ]]
    
    print(user_data)


    result = model.predict(user_data)

    print(result)

    target = ['not promoted', 'promoted']

    print(f"prediction = {target[result[0]]}")


    return target[result[0]]

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=8080)

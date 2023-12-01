from flask import Flask, render_template, request
from keras.models import load_model
from model import translate_sentence

app = Flask(__name__)

# Load the model
seq2seq_model = load_model('seq2seq_model.h5')

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/team')
def team():
    return render_template("team.html")

@app.route('/translate', methods=['POST'])
def predict():
    text = request.form.get('text')
    if text:
        result = translate_sentence(text)
        return render_template('result.html', text=text, result=result)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
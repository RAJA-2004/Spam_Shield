from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

#========================loading the save files==================================================
model = pickle.load(open('logistic_regression.pkl','rb'))
feature_extraction = pickle.load(open('feature_extraction.pkl','rb'))


def predict_mail(input_text):
    input_user_mail  = [input_text]
    input_data_features = feature_extraction.transform(input_user_mail)
    prediction = model.predict(input_data_features)
    return prediction


@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/classifier', methods=['GET', 'POST'])
def analyze_mail():
    if request.method == 'POST':
        mail = request.form.get('mail')
        predicted_mail = predict_mail(input_text=mail)
        return render_template('index.html', classify=predicted_mail, mail_text=mail)

    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    return analyze_mail()

if __name__ == '__main__':
    # Use PORT environment variable for deployment
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

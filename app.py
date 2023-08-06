from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        df = pd.read_csv("spam.csv", encoding="latin-1")
        df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
        # Features and Labels
        df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
        X = df['v2']
        y = df['label']
        cv = CountVectorizer()
        X = cv.fit_transform(X)  # Fit the Data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        clf = MultinomialNB()
        clf.fit(X_train, y_train)

        clf_score = clf.score(X_test, y_test)

        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)

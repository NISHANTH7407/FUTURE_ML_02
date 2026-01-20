
from flask import Flask, render_template, request
import pickle, re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

def assign_priority(text):
    text = text.lower()
    if any(w in text for w in ['urgent','error','failed','down','not working']):
        return 'High'
    elif any(w in text for w in ['reset','request','access','update']):
        return 'Medium'
    else:
        return 'Low'

app = Flask(__name__)
model = pickle.load(open('model/category_model.pkl','rb'))
vectorizer = pickle.load(open('model/tfidf_vectorizer.pkl','rb'))

@app.route('/', methods=['GET','POST'])
def index():
    category = priority = None
    if request.method == 'POST':
        text = request.form['ticket']
        vec = vectorizer.transform([clean_text(text)])
        category = model.predict(vec)[0]
        priority = assign_priority(text)
    return render_template('index.html', category=category, priority=priority)

if __name__ == '__main__':
    app.run(debug=True)

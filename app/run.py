import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify
from sqlalchemy import create_engine
import ktrain


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse_split.db')
df = pd.read_sql_table('DisasterResponse_split', engine)

# load model
#model = joblib.load("../models/bert_finetuned_model")
#preproc = joblib.load("../models/bert_finetuned_mode.preproc")

# Instantiate predictor object
loaded_model = ktrain.load_predictor('../models/bert_finetuned_model')

@app.route('/')
@app.route('/index')
def index():
    # render web page 
    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    #classification_labels = model.predict([query])[0]
    #classification_results = dict(zip(df.columns[4:], classification_labels))

    target_categories = df.columns[5:]
    probabilities = loaded_model.predict_proba([query])[0]
    classification_results = {target_category: prob for target_category, prob in zip(target_categories, probabilities)}

    # renders go.html  
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=4444, debug=True)


if __name__ == '__main__':
    main()
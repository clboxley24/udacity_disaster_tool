# Disaster Response Pipeline Project

This project creates a database and classification model. The classification model is wrapped into a Flask app capable of categorizing disaster messages in 36 categories.

### Dependencies:
- Python 3.5+
- NumPy
- SciPy
- Pandas
- Sklearn
- NLTK
- SQLalchemy
- Flask
- Plotly

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

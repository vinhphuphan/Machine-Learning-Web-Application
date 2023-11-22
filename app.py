from flask import Flask, render_template, request,redirect 
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import spacy
from spacy import displacy
from flaskext.markdown import Markdown
nltk.download('vader_lexicon')

UPLOAD_FOLDER = "./static/user_input/"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model("second_model.h5")
nlp = spacy.load('en_core_web_sm')
Markdown(app)

# routes
@app.route("/")
def index():
    # Determine the active section here
    active_section = "home"  # Replace this with your logic to determine the active section

    # Pass the active section to the template
    return render_template("index.html", active_section=active_section)

@app.route("/", methods=['POST'])
def handle_form_submission():
    # Determine the active section from the request
    active_section = "home"
    # Check which form was submitted based on the form name attribute
    if "image-classification-form" in request.form:
        # Delete existing files in the directory
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
                    
        prediction = None
        if request.method == 'POST':
            # check if the post request has the file part
            if request.files["imagefile"] == None or request.files["imagefile"].filename == '':
                return redirect(request.url)

            img = request.files['imagefile']
            filename = img.filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(img_path)
                    
            if img_path:
                img = image.load_img(img_path, target_size=(160, 160))
                img = image.img_to_array(img)
                img = np.reshape(img, [1, 160 , 160, 3])
                # Make predictions
                prediction = model.predict(img)
                prediction = tf.nn.sigmoid(prediction)
                # Interpret the predictions
                if prediction < 0.5:
                    prediction =  "Cat"
                else:
                    prediction = "Dog"

        return render_template("index.html", prediction=prediction, img_path=img_path, active_section="image-classifier")
    
    elif "sentiment-analysis-form" in request.form:
        message = None
        if request.method == "POST":
            input_text = request.form.get("sentiment-analysis-form")
            analyzer = SentimentIntensityAnalyzer()
            score = analyzer.polarity_scores(str(input_text))
            if score["compound"] < 0:
                message = "Negative"
            elif score["compound"] > 0:
                message = "Positive"
            else :
                message = "Neutral"
        return render_template("index.html", message=message, text = str(input_text), active_section="sentiment-analysis")
    
    elif "NER-form" in request.form:
        raw_text = None
        NER_result = None
        if request.method == "POST":
            raw_text = request.form.get("NER-form")
            doc = nlp(raw_text)
            html = displacy.render(doc, style="ent")
            html = html.replace("\n\n", "\n")
            NER_result = html

        return render_template("index.html", raw_text=raw_text, NER_result=NER_result, active_section="NER")

    return render_template("index.html")

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port=5000, debug=True)
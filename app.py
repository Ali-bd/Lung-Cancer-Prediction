from flask import Flask, render_template, request
import joblib
import pdfplumber  # Use pdfplumber for PDF extraction

app = Flask(__name__)

# Load your model
model = joblib.load('svm_model.pkl')

# Function to extract data from PDF using pdfplumber
def extract_data_from_pdf(pdf_path):
    features = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            # Logic to extract values from the text (adjust this based on your PDF format)
            for line in text.splitlines():
                if "KCNA4" in line:
                    features['KCNA4'] = float(line.split(":")[1].strip())
                elif "GRIA1" in line:
                    features['GRIA1'] = float(line.split(":")[1].strip())
                elif "SCN7A" in line:
                    features['SCN7A'] = float(line.split(":")[1].strip())
                elif "KCNQ3" in line:
                    features['KCNQ3'] = float(line.split(":")[1].strip())
                elif "SPTBN1" in line:
                    features['SPTBN1'] = float(line.split(":")[1].strip())
                elif "SDC1" in line:
                    features['SDC1'] = float(line.split(":")[1].strip())
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_method = request.form['input_method']

    if input_method == 'manual':
        # Get data from manual input
        kcna4 = float(request.form['KCNA4'])
        gria1 = float(request.form['GRIA1'])
        scn7a = float(request.form['SCN7A'])
        kcnq3 = float(request.form['KCNQ3'])
        sptbn1 = float(request.form['SPTBN1'])
        sdc1 = float(request.form['SDC1'])
        data = [[kcna4, gria1, scn7a, kcnq3, sptbn1, sdc1]]

    elif input_method == 'pdf':
        # Get data from the uploaded PDF
        pdf_file = request.files['pdf']
        pdf_path = "uploaded_pdf.pdf"
        pdf_path = "uploaded_pdf_files/"+pdf_path
        pdf_file.save(pdf_path)
        features = extract_data_from_pdf(pdf_path)

        # Ensure that all required features are extracted from the PDF
        data = [[
            features.get('KCNA4', 0),  # Default value is 0 if not found
            features.get('GRIA1', 0),
            features.get('SCN7A', 0),
            features.get('KCNQ3', 0),
            features.get('SPTBN1', 0),
            features.get('SDC1', 0)
        ]]

    # Make prediction using the model
    prediction = model.predict(data)
    result = 'Positive' if prediction[0] == 1 else 'Negative'

    # Render the prediction page with results
    return render_template('prediction.html',
                           kcna4=data[0][0],
                           gria1=data[0][1],
                           scn7a=data[0][2],
                           kcnq3=data[0][3],
                           sptbn1=data[0][4],
                           sdc1=data[0][5],
                           result=result)

if __name__ == "__main__":
    app.run(debug=True)

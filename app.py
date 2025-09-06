import streamlit as st
import nltk
import re
import pickle

# Download required NLTK packages
nltk.download('punkt')
nltk.download('stopwords')

# Load trained models
try:
    clf = pickle.load(open('clf.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Fixed variable name
except FileNotFoundError:
    st.error("Error: Model files ('clf.pkl' and 'tfidf.pkl') not found. Ensure they are in the correct directory.")
    st.stop()


# Function to clean resume text
def clean_Resume(resume_text):
    clean_text = re.sub(r'http\S+\s', ' ', resume_text)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+\s', ' ', clean_text)
    clean_text = re.sub(r'@\S+', ' ', clean_text)
    clean_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^-{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)  # Fixed encoding issue
    clean_text = re.sub(r'\s+', ' ', clean_text)  # Remove extra spaces
    return clean_text.strip()


# Web app interface
def main():
    st.title("Resume Screening App")

    uploaded_file = st.file_uploader("Upload Resume", type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')  # Try decoding as UTF-8
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')  # Fallback to Latin-1 if UTF-8 fails

        cleaned_resume = clean_Resume(resume_text)

        # Transform text using TF-IDF
        input_features = tfidf.transform([cleaned_resume])  # Fixed TF-IDF transformation

        # Predict category
        prediction_id = clf.predict(input_features)[0]

        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and Fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate"
        }

        # Get the predicted category name
        category_name = category_mapping.get(prediction_id, "Unknown")

        # Display prediction
        st.success(f"Predicted Category: **{category_name}**")


# Run Streamlit app
if __name__ == "__main__":
    main()

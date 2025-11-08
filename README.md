🩺 Heart Disease Prediction and AI Recommendation System
💡 Overview

This project is an AI-powered system that predicts the risk of heart disease using multiple machine learning models and offers personalized health recommendations such as diet plans, exercise routines, and lifestyle changes.

The goal is to help people understand their heart health risk early through data-driven insights — combining healthcare and technology to promote smarter well-being decisions.

⚙️ Tech Stack

Language: Python

Frontend: Streamlit

Libraries & Tools:

Pandas

NumPy

Scikit-learn

XGBoost

Matplotlib / Seaborn

🧠 Machine Learning Models

To achieve the best accuracy, the project compares several models:

Logistic Regression

Support Vector Machine (SVM)

Random Forest Classifier

XGBoost Classifier

Each model was trained, tested, and evaluated on accuracy, precision, recall, and F1-score.
The app can automatically select or allow the user to choose the best-performing model for prediction.

🧾 Dataset

The dataset includes key health indicators such as:

Age

Gender

Cholesterol levels

Blood Pressure

Chest Pain Type

Blood Sugar

ECG Results

Maximum Heart Rate, etc.

All data is cleaned, normalized, and prepared for model training.
You can find the dataset inside the Dataset/ folder.

🎨 Frontend (Streamlit App)

The Streamlit web app provides an easy-to-use interface where users can:

Enter their medical details

Instantly get heart disease predictions

Receive AI-based health advice

Explore model accuracy and performance visuals

To launch the app:

streamlit run frontend/app.py

📂 Project Structure
Heart-Disease-Predictor/
│
├── ColabFile/           # Jupyter Notebook for model development
├── Dataset/             # Heart disease dataset
├── frontend/            # Streamlit interface files
├── requirements.txt     # Dependencies
└── README.md            # Project details

🚀 How to Run

Clone the Repository

git clone https://github.com/24MCA003/Heart-Disease-Predictor.git
cd Heart-Disease-Predictor


Install Dependencies

pip install -r requirements.txt


Run the Streamlit App

streamlit run frontend/app.py

🧩 Key Features

✅ Predicts heart disease risk using multiple ML models
✅ Gives personalized exercise, diet, and lifestyle tips
✅ Clean, modular, and easy-to-understand codebase
✅ Visual performance metrics for each model
✅ Streamlit-based interactive user interface

📈 Future Enhancements

Add deep learning models for higher accuracy

Include user authentication and health history tracking

Deploy the project online for real-time public use

👨‍💻 Developer

Nikesh Barodiya
Postgraduate Student, Nirma University
📧 24mca003@nirmauni.ac.in

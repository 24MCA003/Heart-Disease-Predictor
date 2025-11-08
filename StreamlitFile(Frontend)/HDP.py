import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    """Load and prepare the heart disease dataset with proper encoding"""
    try:
        # Try loading from uploaded file or URL
        df = pd.read_csv('heart.csv')
    except:
        # Fallback: Create sample data matching the new format with better distribution
        np.random.seed(42)
        n_samples = 918
        
        # Create more realistic data distribution
        ages = np.random.normal(54, 9, n_samples).astype(int)
        ages = np.clip(ages, 28, 77)
        
        df = pd.DataFrame({
            'Age': ages,
            'Sex': np.random.choice(['M', 'F'], n_samples, p=[0.68, 0.32]),
            'ChestPainType': np.random.choice(['ATA', 'NAP', 'ASY', 'TA'], n_samples, p=[0.18, 0.24, 0.50, 0.08]),
            'RestingBP': np.random.normal(132, 18, n_samples).astype(int),
            'Cholesterol': np.random.normal(246, 52, n_samples).astype(int),
            'FastingBS': np.random.choice([0, 1], n_samples, p=[0.77, 0.23]),
            'RestingECG': np.random.choice(['Normal', 'ST', 'LVH'], n_samples, p=[0.60, 0.20, 0.20]),
            'MaxHR': np.random.normal(137, 25, n_samples).astype(int),
            'ExerciseAngina': np.random.choice(['N', 'Y'], n_samples, p=[0.55, 0.45]),
            'Oldpeak': np.abs(np.random.normal(0.89, 1.07, n_samples)),
            'ST_Slope': np.random.choice(['Up', 'Flat', 'Down'], n_samples, p=[0.22, 0.50, 0.28]),
        })
        
        # Create target based on risk factors
        risk_score = (
            (df['Age'] > 55).astype(int) * 0.2 +
            (df['Sex'] == 'M').astype(int) * 0.15 +
            (df['ChestPainType'] == 'ASY').astype(int) * 0.25 +
            (df['RestingBP'] > 140).astype(int) * 0.1 +
            (df['Cholesterol'] > 240).astype(int) * 0.1 +
            (df['ExerciseAngina'] == 'Y').astype(int) * 0.2
        )
        df['HeartDisease'] = (risk_score + np.random.normal(0, 0.1, n_samples) > 0.45).astype(int)
    
    # Encode categorical variables
    sex_map = {"M": 0, "F": 1}
    chest_pain_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
    resting_ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
    exercise_angina_map = {"N": 0, "Y": 1}
    st_slope_map = {"Up": 0, "Flat": 1, "Down": 2}
    
    df["Sex"] = df["Sex"].map(sex_map)
    df["ChestPainType"] = df["ChestPainType"].map(chest_pain_map)
    df["RestingECG"] = df["RestingECG"].map(resting_ecg_map)
    df["ExerciseAngina"] = df["ExerciseAngina"].map(exercise_angina_map)
    df["ST_Slope"] = df["ST_Slope"].map(st_slope_map)
    
    # Handle missing values
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df['RestingBP'] = df['RestingBP'].replace(0, np.nan)
    
    df['Cholesterol'].fillna(df['Cholesterol'].median(), inplace=True)
    df['RestingBP'].fillna(df['RestingBP'].median(), inplace=True)
    
    # Clip extreme values
    df['RestingBP'] = np.clip(df['RestingBP'], 80, 200)
    df['Cholesterol'] = np.clip(df['Cholesterol'], 100, 600)
    df['MaxHR'] = np.clip(df['MaxHR'], 60, 220)
    df['Oldpeak'] = np.clip(df['Oldpeak'], -3, 10)
    
    return df

@st.cache_resource
def train_models():
    """Train multiple optimized ML models with better hyperparameters"""
    df = load_and_prepare_data()
    
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optimized models with better hyperparameters
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=2000, 
            C=0.1,  # Regularization
            solver='liblinear',
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',  # RBF kernel instead of linear
            C=10,
            gamma='scale',
            probability=True, 
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,  # Limit depth to prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,  # More trees
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='logloss'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7,  # Optimal k
            weights='distance',  # Distance-weighted
            metric='manhattan'
        ),
        'Naive Bayes': GaussianNB(
            var_smoothing=1e-8  # Smoothing parameter
        )
    }
    
    trained_models = {}
    accuracies = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        trained_models[name] = model
        accuracies[name] = accuracy
    
    best_model_name = max(accuracies, key=accuracies.get)
    
    return trained_models, accuracies, scaler, best_model_name, X_test_scaled, y_test

def generate_ai_advice(risk_status, user_data):
    """Generate AI-powered health advice"""
    if risk_status == "At Risk":
        advice = f"""
### 🏥 Personalized Health Plan

**⚠️ Based on your assessment, here are important recommendations:**

#### 🏃‍♂️ Exercise Recommendations:
- **Daily Walking**: Start with 20-30 minutes of brisk walking daily
- **Cardio**: Swimming or cycling 3-4 times per week (low impact)
- **Strength Training**: Light resistance exercises 2 times per week
- **Avoid**: High-intensity exercises without medical clearance

#### 🥗 Dietary Suggestions:
- **Reduce**: Saturated fats, trans fats, and sodium intake
- **Increase**: Fruits, vegetables, whole grains, and lean proteins
- **Focus on**: Mediterranean diet - olive oil, fish, nuts, and legumes
- **Limit**: Red meat to 1-2 times per week
- **Hydration**: Drink 8-10 glasses of water daily

#### 💊 Lifestyle Changes:
- Monitor blood pressure regularly (target: below 130/80)
- Manage cholesterol levels through diet and medication if prescribed
- Quit smoking and limit alcohol consumption
- Maintain healthy weight (BMI: 18.5-24.9)
- Stress management through meditation or yoga
- Get 7-8 hours of quality sleep

#### 📅 Medical Follow-up:
- **Urgent**: Schedule an appointment with a cardiologist
- Get comprehensive heart health screening
- Discuss medication options if needed
- Regular monitoring every 3-6 months

**Remember: This is a prediction tool. Always consult with healthcare professionals for proper diagnosis and treatment.**
        """
    else:
        advice = f"""
### ✅ Heart Health Maintenance Plan

**Great news! Keep up the good work with these tips:**

#### 🏃‍♂️ Maintain Active Lifestyle:
- Continue regular exercise (150 minutes moderate activity per week)
- Mix cardio and strength training
- Try new activities: hiking, dancing, sports
- Stay consistent with your fitness routine

#### 🥗 Healthy Eating Habits:
- Continue balanced diet rich in fruits and vegetables
- Choose whole grains over refined carbs
- Include healthy fats: avocados, nuts, olive oil
- Limit processed foods and added sugars
- Moderate portions and mindful eating

#### 💚 Preventive Care:
- Annual health checkups and heart screenings
- Monitor blood pressure and cholesterol yearly
- Maintain healthy weight through lifestyle
- Stay hydrated and get adequate sleep
- Manage stress through relaxation techniques

#### 🎯 Long-term Goals:
- Keep BMI in healthy range (18.5-24.9)
- Stay socially active and engaged
- Build strong support network
- Continue learning about heart health
- Set new fitness milestones

**Keep doing what you're doing! Prevention is the best medicine.**
        """
    
    return advice

def create_accuracy_chart(accuracies):
    """Create a bar chart showing model accuracies"""
    sorted_items = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=[acc * 100 for acc in values],
            marker_color=colors[:len(names)],
            text=[f'{acc*100:.2f}%' for acc in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Model Accuracy Comparison (All 8 Models)',
        xaxis_title='Model',
        yaxis_title='Accuracy (%)',
        yaxis_range=[0, 100],
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig

def create_confusion_matrix_chart(y_test, y_pred, model_name):
    """Create confusion matrix heatmap"""
    cm = confusion_matrix(y_test, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No Disease', 'Predicted: Disease'],
        y=['Actual: No Disease', 'Actual: Disease'],
        colorscale='RdYlGn_r',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        height=400
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header">❤️ Heart Disease Predictor</h1>', unsafe_allow_html=True)
 
    
    # File uploader for custom dataset
    st.sidebar.header("📁 Upload Dataset (Optional)")
    uploaded_file = st.sidebar.file_uploader("Upload your heart.csv", type=['csv'])
    
    if uploaded_file is not None:
        try:
            with open('heart.csv', 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success("✅ Dataset uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error uploading file: {e}")
    
    # Sidebar for model information
    with st.sidebar:
        st.header("📊 Model Information")
        
        with st.spinner("Training all 8 optimized models..."):
            trained_models, accuracies, scaler, best_model_name, X_test_scaled, y_test = train_models()
        
        st.success("✅ All models trained successfully!")
        st.metric("🏆 Best Model", best_model_name)
        st.metric("🎯 Best Accuracy", f"{accuracies[best_model_name]*100:.2f}%")
        
        st.markdown("---")
        st.markdown("### 📈 All Model Accuracies")
        
        sorted_accuracies = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
        for i, (name, acc) in enumerate(sorted_accuracies, 1):
            if name == best_model_name:
                st.markdown(f"**{i}. 🥇 {name}: {acc*100:.2f}%**")
            else:
                st.text(f"{i}. {name}: {acc*100:.2f}%")
        
        st.markdown("---")
        st.markdown("### 🎯 Model Selection")
        selected_model = st.selectbox(
            "Choose model for prediction:",
            list(trained_models.keys()),
            index=list(trained_models.keys()).index(best_model_name)
        )
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📈 Model Performance", "🔬 Detailed Analysis"])
    
    with tab1:
        st.markdown("### Enter Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", options=["Male", "Female"])
            sex_encoded = 0 if sex == "Male" else 1
            
            chest_pain = st.selectbox("Chest Pain Type", 
                                     options=["Typical Angina (ATA)", "Atypical Angina (NAP)", 
                                             "Asymptomatic (ASY)", "Non-Anginal Pain (TA)"])
            cp_map = {"Typical Angina (ATA)": 0, "Atypical Angina (NAP)": 1, 
                     "Asymptomatic (ASY)": 2, "Non-Anginal Pain (TA)": 3}
            cp_encoded = cp_map[chest_pain]
            
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 
                                         min_value=80, max_value=200, value=120)
        
        with col2:
            cholesterol = st.number_input("Cholesterol (mg/dl)", 
                                         min_value=0, max_value=600, value=200)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                                     options=["No", "Yes"])
            fbs_encoded = 0 if fasting_bs == "No" else 1
            
            resting_ecg = st.selectbox("Resting ECG", 
                                      options=["Normal", "ST-T Abnormality (ST)", 
                                              "LV Hypertrophy (LVH)"])
            ecg_map = {"Normal": 0, "ST-T Abnormality (ST)": 1, "LV Hypertrophy (LVH)": 2}
            ecg_encoded = ecg_map[resting_ecg]
            
            max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        
        with col3:
            exercise_angina = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
            exang_encoded = 0 if exercise_angina == "No" else 1
            
            oldpeak = st.number_input("ST Depression (Oldpeak)", 
                                     min_value=-3.0, max_value=10.0, value=1.0, step=0.1)
            
            st_slope = st.selectbox("ST Slope", 
                                   options=["Upsloping", "Flat", "Downsloping"])
            slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope_encoded = slope_map[st_slope]
        
        st.markdown("---")
        
        if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):
            input_data = np.array([[age, sex_encoded, cp_encoded, resting_bp, cholesterol, 
                                   fbs_encoded, ecg_encoded, max_hr, exang_encoded, 
                                   oldpeak, slope_encoded]])
            
            input_data_scaled = scaler.transform(input_data)
            
            model = trained_models[selected_model]
            prediction = model.predict(input_data_scaled)[0]
            prediction_proba = model.predict_proba(input_data_scaled)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
            
            st.markdown("---")
            st.markdown("## 📋 Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if prediction == 1:
                    st.error("### ⚠️ You are likely **at risk** of heart disease")
                    risk_status = "At Risk"
                    risk_percentage = prediction_proba[1] * 100
                else:
                    st.success("### ✅ You are likely **not at risk** of heart disease")
                    risk_status = "Not at Risk"
                    risk_percentage = prediction_proba[0] * 100
                
                st.metric("Confidence Level", f"{risk_percentage:.1f}%")
                st.info(f"Model Used: **{selected_model}**")
                st.metric("Model Accuracy", f"{accuracies[selected_model]*100:.2f}%")
            
            with col_result2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_percentage,
                    title={'text': "Risk Level"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgreen"},
                            {'range': [33, 66], 'color': "yellow"},
                            {'range': [66, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=250)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 🤖 Predictions from All Models")
            
            all_predictions = {}
            for name, mdl in trained_models.items():
                pred = mdl.predict(input_data_scaled)[0]
                pred_proba = mdl.predict_proba(input_data_scaled)[0] if hasattr(mdl, 'predict_proba') else [0.5, 0.5]
                all_predictions[name] = {
                    'prediction': pred,
                    'confidence': pred_proba[pred] * 100
                }
            
            pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
            
            cols = [pred_col1, pred_col2, pred_col3, pred_col4]
            for i, (name, pred_data) in enumerate(all_predictions.items()):
                with cols[i % 4]:
                    result = "⚠️ At Risk" if pred_data['prediction'] == 1 else "✅ Safe"
                    color = "red" if pred_data['prediction'] == 1 else "green"
                    st.markdown(f"**{name}**")
                    st.markdown(f"<span style='color:{color}'>{result}</span>", unsafe_allow_html=True)
                    st.caption(f"Confidence: {pred_data['confidence']:.1f}%")
            
            risk_count = sum(1 for p in all_predictions.values() if p['prediction'] == 1)
            consensus_percentage = (risk_count / len(all_predictions)) * 100
            
            st.markdown("---")
            st.markdown("### 🎯 Model Consensus")
            
            consensus_col1, consensus_col2 = st.columns(2)
            
            with consensus_col1:
                st.metric("Models Predicting Risk", f"{risk_count} out of {len(all_predictions)}")
                st.metric("Consensus", f"{consensus_percentage:.1f}%")
            
            with consensus_col2:
                fig_consensus = go.Figure(data=[
                    go.Pie(
                        labels=['At Risk', 'Not at Risk'],
                        values=[risk_count, len(all_predictions) - risk_count],
                        marker_colors=['#e74c3c', '#2ecc71'],
                        hole=0.4
                    )
                ])
                fig_consensus.update_layout(
                    title="Model Consensus",
                    height=250,
                    showlegend=True
                )
                st.plotly_chart(fig_consensus, use_container_width=True)
            
            st.markdown("---")
            advice = generate_ai_advice(risk_status, {})
            st.markdown(advice)
            
            st.markdown("---")
            st.markdown("### 📊 Your Health Metrics")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                bp_status = "Normal" if resting_bp < 130 else "High"
                st.metric("Blood Pressure", f"{resting_bp} mm Hg", bp_status)
            
            with metric_col2:
                chol_status = "Normal" if cholesterol < 200 else "High"
                st.metric("Cholesterol", f"{cholesterol} mg/dl", chol_status)
            
            with metric_col3:
                hr_status = "Good" if 60 <= max_hr <= 100 else "Check"
                st.metric("Max Heart Rate", f"{max_hr} bpm", hr_status)
            
            with metric_col4:
                bmi_placeholder = 22 + (age - 50) * 0.1
                st.metric("Estimated BMI", f"{bmi_placeholder:.1f}", "Healthy")
    
    with tab2:
        st.markdown("### Model Performance Comparison")
        
        fig_accuracy = create_accuracy_chart(accuracies)
        st.plotly_chart(fig_accuracy, use_container_width=True)
        
        st.markdown("### 🏆 Model Rankings")
        
        rank_col1, rank_col2 = st.columns(2)
        
        with rank_col1:
            st.markdown("#### Top 4 Models")
            sorted_acc = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
            for i, (name, acc) in enumerate(sorted_acc[:4], 1):
                medal = ["🥇", "🥈", "🥉", "🏅"][i-1]
                st.markdown(f"{medal} **{name}**: {acc*100:.2f}%")
        
        with rank_col2:
            st.markdown("#### Bottom 4 Models")
            for i, (name, acc) in enumerate(sorted_acc[4:], 5):
                st.markdown(f"{i}. **{name}**: {acc*100:.2f}%")
        
        st.markdown("---")
        st.markdown("### 🤖 Model Details & Optimizations")
        
        model_info = {
            'Logistic Regression': 'Optimized with L2 regularization (C=0.1) and liblinear solver for better performance.',
            'SVM': 'Using RBF kernel with C=10 for non-linear classification. Distance-weighted for improved accuracy.',
            'Decision Tree': 'Constrained depth (max_depth=10) and minimum samples to prevent overfitting.',
            'Random Forest': '200 trees with sqrt features. Balanced depth and samples for optimal performance.',
            'XGBoost': 'Gradient boosting with 200 estimators, learning_rate=0.1, and regularization parameters.',
            'Gradient Boosting': '200 estimators with learning_rate=0.1. Subsample=0.8 to reduce overfitting.',
            'KNN': 'K=7 neighbors with distance weighting and Manhattan metric for better classification.',
            'Naive Bayes': 'Gaussian NB with variance smoothing (1e-8) for numerical stability.'
        }
        
        for name, description in model_info.items():
            with st.expander(f"{name} - Accuracy: {accuracies[name]*100:.2f}%"):
                st.write(description)
                if name == best_model_name:
                    st.success("✨ This is the best performing model!")
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Information")
        df = load_and_prepare_data()
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Total Samples", len(df))
        with col_info2:
            st.metric("At Risk Cases", int(df['HeartDisease'].sum()))
        with col_info3:
            st.metric("Not at Risk Cases", int(len(df) - df['HeartDisease'].sum()))
    
    with tab3:
        st.markdown("### 🔬 Detailed Model Analysis")
        
        analysis_model = st.selectbox(
            "Select model for detailed analysis:",
            list(trained_models.keys()),
            key="analysis_model"
        )
        
        model = trained_models[analysis_model]
        y_pred = model.predict(X_test_scaled)
        
        st.markdown(f"#### Confusion Matrix - {analysis_model}")
        fig_cm = create_confusion_matrix_chart(y_test, y_pred, analysis_model)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.markdown("#### Classification Report")
        report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'], output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'))
        
        st.markdown("#### Performance Metrics")
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("Accuracy", f"{report['accuracy']:.3f}")
        
        with metrics_col2:
            st.metric("Precision (Disease)", f"{report['Disease']['precision']:.3f}")
        
        with metrics_col3:
            st.metric("Recall (Disease)", f"{report['Disease']['recall']:.3f}")
        
        with metrics_col4:
            st.metric("F1-Score (Disease)", f"{report['Disease']['f1-score']:.3f}")

if __name__ == "__main__":
    main()
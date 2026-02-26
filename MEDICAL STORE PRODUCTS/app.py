import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the App
st.title("Medical Data Analysis - Hospital Patient Dataset")
st.write("This application analyzes patient medical data including blood pressure and disease categories.")

# Generate Synthetic Medical Data
def generate_data():
    np.random.seed(42)
    data = {
        'patient_id': range(1, 21),
        'patient_name': [f'Patient {i}' for i in range(1, 21)],
        'disease_type': np.random.choice(['Diabetes', 'Hypertension', 'Cardiac', 'Asthma'], 20),
        'blood_pressure': np.random.normal(loc=120, scale=15, size=20).astype(int),
        'admission_date': pd.date_range(start='2023-01-01', periods=20, freq='D')
    }
    return pd.DataFrame(data)

medical_data = generate_data()

# Display Data
st.subheader("Patient Medical Data")
st.dataframe(medical_data)

# -------------------------
# Descriptive Statistics
# -------------------------
st.subheader("Descriptive Statistics (Blood Pressure)")

descriptive_stats = medical_data['blood_pressure'].describe()
st.write(descriptive_stats)

mean_bp = medical_data['blood_pressure'].mean()
median_bp = medical_data['blood_pressure'].median()
mode_bp = medical_data['blood_pressure'].mode()[0]

st.write(f"Mean Blood Pressure: {mean_bp}")
st.write(f"Median Blood Pressure: {median_bp}")
st.write(f"Mode Blood Pressure: {mode_bp}")

# Group Statistics by Disease
disease_stats = medical_data.groupby('disease_type')['blood_pressure'].agg(['mean', 'std', 'min', 'max']).reset_index()
disease_stats.columns = ['Disease Type', 'Average BP', 'Std Dev BP', 'Min BP', 'Max BP']

st.subheader("Disease-wise Blood Pressure Statistics")
st.dataframe(disease_stats)

# -------------------------
# Inferential Statistics
# -------------------------
confidence_level = 0.95
degrees_freedom = len(medical_data['blood_pressure']) - 1
sample_mean = mean_bp
sample_standard_error = medical_data['blood_pressure'].std() / np.sqrt(len(medical_data['blood_pressure']))

t_score = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
margin_of_error = t_score * sample_standard_error
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

st.subheader("95% Confidence Interval for Mean Blood Pressure")
st.write(confidence_interval)

# Hypothesis Testing
# H0: Mean BP = 120 (Normal Average BP)
t_statistic, p_value = stats.ttest_1samp(medical_data['blood_pressure'], 120)

st.subheader("Hypothesis Testing (t-test for Mean BP = 120)")
st.write(f"T-statistic: {t_statistic}")
st.write(f"P-value: {p_value}")

if p_value < 0.05:
    st.write("Reject the null hypothesis: Mean blood pressure is significantly different from 120.")
else:
    st.write("Fail to reject the null hypothesis: Mean blood pressure is not significantly different from 120.")

# -------------------------
# Visualizations
# -------------------------
st.subheader("Visualizations")

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(medical_data['blood_pressure'], bins=10, kde=True)
plt.axvline(mean_bp, color='red', linestyle='--', label='Mean')
plt.axvline(median_bp, color='blue', linestyle='--', label='Median')
plt.axvline(mode_bp, color='green', linestyle='--', label='Mode')
plt.title('Distribution of Blood Pressure')
plt.xlabel('Blood Pressure')
plt.ylabel('Frequency')
plt.legend()
st.pyplot(plt)

# Boxplot by Disease
plt.figure(figsize=(10, 6))
sns.boxplot(x='disease_type', y='blood_pressure', data=medical_data)
plt.title('Blood Pressure by Disease Type')
plt.xlabel('Disease Type')
plt.ylabel('Blood Pressure')
st.pyplot(plt)

# Bar Plot for Average BP by Disease
plt.figure(figsize=(10, 6))
sns.barplot(x='Disease Type', y='Average BP', data=disease_stats)
plt.title('Average Blood Pressure by Disease Type')
plt.xlabel('Disease Type')
plt.ylabel('Average BP')
st.pyplot(plt)
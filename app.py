import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Test Result Predictor",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
    .main-title {font-size:2.2rem; font-weight:700; color:#1a5276; margin-bottom:4px;}
    .sub-title  {font-size:1rem; color:#555; margin-bottom:24px;}
    .result-box {
        padding:20px; border-radius:12px; text-align:center;
        font-size:1.6rem; font-weight:700; margin-top:16px;
    }
    .result-normal     {background:#d5f5e3; color:#1e8449;}
    .result-abnormal   {background:#fadbd8; color:#922b21;}
    .result-inconclusive {background:#fef9e7; color:#9a7d0a;}
    .stButton>button   {background:#1a5276; color:white; border-radius:8px; width:100%;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏥 Healthcare Test Result Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Masukkan data pasien untuk memprediksi hasil tes (Normal / Abnormal / Inconclusive)</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def categorize_los(days):
    if 1 <= days <= 3:
        return 'A (1-3 days)'
    elif 4 <= days <= 14:
        return 'B (4-14 days)'
    elif days >= 15:
        return 'C (>=15 days)'
    return 'B (4-14 days)'  # fallback

SCORING_MAP = {
    'Cancer': 10, 'Obesity': 7, 'Diabetes': 6,
    'Hypertension': 4, 'Asthma': 3, 'Arthritis': 2
}

CAT_FEATURES = ['Gender', 'Blood Type', 'Medical Condition', 'Doctor', 'Hospital',
                'Insurance Provider', 'Admission Type', 'Medication']

# ─────────────────────────────────────────────
# Synthetic training data (mirrors notebook logic)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Melatih model, harap tunggu...")
def train_model():
    np.random.seed(42)
    n = 2000

    genders        = ['Male', 'Female']
    blood_types    = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    conditions     = list(SCORING_MAP.keys())
    insurances     = ['Medicare', 'Aetna', 'Blue Cross', 'Cigna', 'United Health']
    admissions     = ['Emergency', 'Elective', 'Urgent']
    medications    = ['Aspirin', 'Ibuprofen', 'Paracetamol', 'Lipitor', 'Metformin', 'Penicillin']
    test_results   = ['Normal', 'Abnormal', 'Inconclusive']

    doctors   = [f'Dr. {n}' for n in ['Smith', 'Jones', 'Lee', 'Patel', 'Brown', 'Garcia', 'White']]
    hospitals = ['City General Hospital', 'St. Mary Medical', 'University Hospital',
                 'Regional Medical Center', 'Community Health Clinic']

    admit_dates     = pd.date_range('2019-01-01', '2023-12-31', periods=n)
    los             = np.random.randint(1, 30, n)
    discharge_dates = admit_dates + pd.to_timedelta(los, unit='D')

    df = pd.DataFrame({
        'Age':               np.random.randint(1, 95, n),
        'Gender':            np.random.choice(genders, n),
        'Blood Type':        np.random.choice(blood_types, n),
        'Medical Condition': np.random.choice(conditions, n),
        'Doctor':            np.random.choice(doctors, n),
        'Hospital':          np.random.choice(hospitals, n),
        'Insurance Provider':np.random.choice(insurances, n),
        'Billing Amount':    np.random.uniform(1000, 80000, n),
        'Admission Type':    np.random.choice(admissions, n),
        'Medication':        np.random.choice(medications, n),
        'Date of Admission': admit_dates,
        'Discharge Date':    discharge_dates,
        'Test Results':      np.random.choice(test_results, n, p=[0.4, 0.35, 0.25]),
    })

    # Feature engineering
    df['Days_In_Hospital'] = (df['Discharge Date'] - df['Date of Admission']).dt.days.clip(lower=1)
    df['Cost_Per_Day']     = df['Billing Amount'] / df['Days_In_Hospital']
    df['LOS Category']     = df['Days_In_Hospital'].apply(categorize_los)
    df['Surgical_Risk_Score'] = df['Medical Condition'].map(SCORING_MAP)
    df['Length_of_Stay']   = df['Days_In_Hospital']
    df['Cost_Intensity']   = df['Surgical_Risk_Score'] * df['Length_of_Stay']

    bins   = [0, 14, 24, 65, 120]
    labels = ['Children (00-14 years)', 'Youth (15-24 years)',
              'Adults (25-64 years)', 'Seniors (65 years and over)']
    df['Age Category'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)

    # Encoders
    le     = LabelEncoder()
    le_los = LabelEncoder()
    le_age = LabelEncoder()

    y = le.fit_transform(df['Test Results'])

    X = df.drop(columns=['Test Results', 'Date of Admission', 'Discharge Date'])
    X['LOS Category']  = le_los.fit_transform(X['LOS Category'])
    X['Age Category']  = le_age.fit_transform(X['Age Category'].astype(str))

    t_enc = TargetEncoder(target_type='multiclass', random_state=42)
    cat_encoded = t_enc.fit_transform(X[CAT_FEATURES], y)

    n_classes = len(le.classes_)
    encoded_col_names = [f'{f}_encoded_class_{i}' for f in CAT_FEATURES for i in range(n_classes)]
    cat_df = pd.DataFrame(cat_encoded, columns=encoded_col_names, index=X.index)

    X_enc = X.drop(columns=CAT_FEATURES)
    X_enc = pd.concat([X_enc, cat_df], axis=1)

    # Model
    best_xgb = XGBClassifier(learning_rate=0.05, max_depth=5, n_estimators=50, random_state=42)
    best_rf  = RandomForestClassifier(max_depth=15, min_samples_split=2, n_estimators=100, random_state=42)
    best_hgb = HistGradientBoostingClassifier(learning_rate=0.01, max_depth=5, max_iter=100, random_state=42)

    model = StackingClassifier(
        estimators=[('xgb', best_xgb), ('rf', best_rf), ('hgb', best_hgb)],
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=False
    )
    model.fit(X_enc, y)

    return model, le, le_los, le_age, t_enc, X_enc.columns.tolist(), n_classes

model, le, le_los, le_age, t_enc, feature_cols, n_classes = train_model()

# ─────────────────────────────────────────────
# Prediction function
# ─────────────────────────────────────────────
def predict(inputs: dict):
    df = pd.DataFrame([inputs])
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date']    = pd.to_datetime(df['Discharge Date'])
    df['Days_In_Hospital']  = (df['Discharge Date'] - df['Date of Admission']).dt.days.clip(lower=1)
    df['Cost_Per_Day']      = df['Billing Amount'] / df['Days_In_Hospital']
    df['LOS Category']      = df['Days_In_Hospital'].apply(categorize_los)
    df['LOS Category']      = le_los.transform(df['LOS Category'])
    df['Surgical_Risk_Score'] = df['Medical Condition'].map(SCORING_MAP).fillna(0)
    df['Length_of_Stay']    = df['Days_In_Hospital']
    df['Cost_Intensity']    = df['Surgical_Risk_Score'] * df['Length_of_Stay']

    bins   = [0, 14, 24, 65, 120]
    labels = ['Children (00-14 years)', 'Youth (15-24 years)',
              'Adults (25-64 years)', 'Seniors (65 years and over)']
    df['Age Category'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True).astype(str)
    df['Age Category'] = le_age.transform(df['Age Category'])

    cat_encoded = t_enc.transform(df[CAT_FEATURES])
    encoded_col_names = [f'{f}_encoded_class_{i}' for f in CAT_FEATURES for i in range(n_classes)]
    cat_df = pd.DataFrame(cat_encoded, columns=encoded_col_names, index=df.index)

    numeric_cols = ['Age', 'Billing Amount', 'Days_In_Hospital', 'Cost_Per_Day',
                    'LOS Category', 'Surgical_Risk_Score', 'Length_of_Stay',
                    'Cost_Intensity', 'Age Category']

    final = pd.DataFrame(columns=feature_cols)
    for col in numeric_cols:
        if col in feature_cols:
            final[col] = df[col].values
    for col in encoded_col_names:
        if col in feature_cols:
            final[col] = cat_df[col].values
    final = final.fillna(0)

    pred_enc  = model.predict(final)
    pred_prob = model.predict_proba(final)
    label     = le.inverse_transform(pred_enc)[0]
    probs     = dict(zip(le.classes_, pred_prob[0]))
    return label, probs

# ─────────────────────────────────────────────
# UI – Input Form
# ─────────────────────────────────────────────
st.markdown("---")
with st.form("prediction_form"):
    st.subheader("📋 Data Pasien")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Informasi Dasar**")
        age    = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=45)
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        blood  = st.selectbox("Golongan Darah", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
        cond   = st.selectbox("Kondisi Medis", list(SCORING_MAP.keys()))

    with col2:
        st.markdown("**Informasi Rawat Inap**")
        admission_type = st.selectbox("Tipe Admisi", ["Emergency", "Elective", "Urgent"])
        admit_date     = st.date_input("Tanggal Masuk", value=pd.Timestamp("2023-01-01"))
        discharge_date = st.date_input("Tanggal Keluar", value=pd.Timestamp("2023-01-08"))
        medication     = st.selectbox("Obat", ["Aspirin", "Ibuprofen", "Paracetamol",
                                                "Lipitor", "Metformin", "Penicillin"])

    with col3:
        st.markdown("**Informasi Lainnya**")
        insurance = st.selectbox("Asuransi", ["Medicare", "Aetna", "Blue Cross", "Cigna", "United Health"])
        billing   = st.number_input("Jumlah Tagihan (Rp/$)", min_value=0.0, value=25000.0, step=500.0)
        doctor    = st.selectbox("Dokter", ["Dr. Smith", "Dr. Jones", "Dr. Lee",
                                             "Dr. Patel", "Dr. Brown", "Dr. Garcia", "Dr. White"])
        hospital  = st.selectbox("Rumah Sakit", ["City General Hospital", "St. Mary Medical",
                                                  "University Hospital", "Regional Medical Center",
                                                  "Community Health Clinic"])

    submitted = st.form_submit_button("🔍 Prediksi Hasil Tes")

# ─────────────────────────────────────────────
# Prediction Output
# ─────────────────────────────────────────────
if submitted:
    if discharge_date <= admit_date:
        st.error("⚠️ Tanggal keluar harus setelah tanggal masuk.")
    else:
        inputs = {
            'Age': age, 'Gender': gender, 'Blood Type': blood,
            'Medical Condition': cond, 'Doctor': doctor, 'Hospital': hospital,
            'Insurance Provider': insurance, 'Billing Amount': billing,
            'Admission Type': admission_type, 'Medication': medication,
            'Date of Admission': str(admit_date), 'Discharge Date': str(discharge_date),
        }

        with st.spinner("Memprediksi..."):
            label, probs = predict(inputs)

        style_map = {
            'Normal': 'result-normal',
            'Abnormal': 'result-abnormal',
            'Inconclusive': 'result-inconclusive',
        }
        icon_map = {'Normal': '✅', 'Abnormal': '⚠️', 'Inconclusive': '🔍'}

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        c1, c2 = st.columns([1, 2])
        with c1:
            css_class = style_map.get(label, 'result-normal')
            st.markdown(
                f'<div class="result-box {css_class}">{icon_map.get(label,"")} {label}</div>',
                unsafe_allow_html=True
            )

        with c2:
            st.markdown("**Probabilitas tiap kelas:**")
            for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
                bar_color = {"Normal": "#1e8449", "Abnormal": "#922b21", "Inconclusive": "#9a7d0a"}.get(cls, "#555")
                st.markdown(f"**{cls}** — {prob*100:.1f}%")
                st.progress(float(prob))

        # Feature summary
        days = (pd.Timestamp(discharge_date) - pd.Timestamp(admit_date)).days
        st.markdown("---")
        st.subheader("📌 Ringkasan Fitur Turunan")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Lama Rawat (hari)", days)
        mc2.metric("Biaya per Hari", f"${billing/max(days,1):,.0f}")
        mc3.metric("Skor Risiko Bedah", SCORING_MAP.get(cond, 0))
        mc4.metric("Cost Intensity", f"{SCORING_MAP.get(cond,0)*days:,}")

st.markdown("---")
st.caption("Model: Stacking Classifier (XGBoost + RandomForest + HistGradientBoosting) | Target: Test Results")

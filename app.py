
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "loan_model.pkl"
DATA_PATH = BASE_DIR / "loan_data_cleaned.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    return data

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

st.title("🏦 Credit Risk Assessment")
st.subheader("Loan Eligibility Prediction System")
st.caption(
    "Machine Learning | Random Forest | Educational Project"
)

total = len(df)
approved = int((df["loan_status"].str.strip() == "Approved").sum())
rejected = int((df["loan_status"].str.strip() == "Rejected").sum())

c1, c2, c3 = st.columns(3)
c1.metric("Total Applications", f"{total:,}")
c2.metric("Approved Applications", f"{approved:,}")
c3.metric("Rejected Applications", f"{rejected:,}")

st.divider()

tab1, tab2, tab3 = st.tabs([
    "🔍 Predict Eligibility",
    "📊 Analytics",
    "📁 Dataset"
])

with tab1:
    st.header("Loan Applicant Details")
    st.write("Enter applicant information to generate a model prediction.")

    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            dependents = st.number_input(
                "Number of Dependents", 0, 10, 2
            )
            education = st.selectbox(
                "Education", ["Graduate", "Not Graduate"]
            )
            self_employed = st.selectbox(
                "Self Employed", ["No", "Yes"]
            )
            income = st.number_input(
                "Annual Income (₹)", 0, 1_000_000_000,
                5_000_000, step=100_000
            )
            cibil = st.number_input(
                "CIBIL Score", 300, 900, 700
            )
            loan_term = st.number_input(
                "Loan Term (Years)", 1, 30, 10
            )

        with col2:
            loan_amount = st.number_input(
                "Loan Amount (₹)", 0, 1_000_000_000,
                10_000_000, step=100_000
            )
            residential = st.number_input(
                "Residential Assets Value (₹)", 0,
                1_000_000_000, 5_000_000, step=100_000
            )
            commercial = st.number_input(
                "Commercial Assets Value (₹)", 0,
                1_000_000_000, 3_000_000, step=100_000
            )
            luxury = st.number_input(
                "Luxury Assets Value (₹)", 0,
                1_000_000_000, 5_000_000, step=100_000
            )
            bank_assets = st.number_input(
                "Bank Assets Value (₹)", 0,
                1_000_000_000, 2_000_000, step=100_000
            )

        submitted = st.form_submit_button(
            "Predict Loan Eligibility",
            type="primary",
            use_container_width=True
        )

    if submitted:
        applicant = pd.DataFrame([{
            "no_of_dependents": dependents,
            "education": education,
            "self_employed": self_employed,
            "income_annum": income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil,
            "residential_assets_value": residential,
            "commercial_assets_value": commercial,
            "luxury_assets_value": luxury,
            "bank_asset_value": bank_assets
        }])

        prediction = model.predict(applicant)[0]
        probabilities = model.predict_proba(applicant)[0]
        classes = list(model.classes_)
        confidence = float(
            probabilities[classes.index(prediction)] * 100
        )

        st.divider()
        if prediction == "Approved":
            st.success("Model Prediction: LOAN APPROVED")
        else:
            st.error("Model Prediction: LOAN REJECTED")

        st.metric("Model Prediction Confidence", f"{confidence:.1f}%")
        st.progress(min(max(confidence / 100, 0.0), 1.0))
        st.caption(
            "Educational model estimate only; not a bank decision "
            "or a guarantee of loan approval."
        )

with tab2:
    st.header("Application Analytics")

    left, right = st.columns(2)

    with left:
        status_counts = (
            df["loan_status"].str.strip()
            .value_counts().rename_axis("Status")
            .reset_index(name="Applications")
        )
        fig = px.pie(
            status_counts, names="Status", values="Applications",
            title="Loan Status Distribution", hole=0.45
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            df, x="cibil_score", color="loan_status",
            title="CIBIL Score Distribution",
            labels={"cibil_score": "CIBIL Score",
                    "loan_status": "Loan Status"},
            barmode="overlay"
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        df, x="loan_status", y="income_annum",
        color="loan_status", title="Annual Income by Loan Status"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Dataset Explorer")
    st.write(f"Dataset size: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="loan_data_cleaned.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "Credit Risk Assessment & Loan Eligibility Prediction | "
    "Academic Demonstration"
)

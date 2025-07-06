import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix

# ========== SETUP ==========
st.set_page_config(layout="wide")
st.title("🛍️ GLAMBASKET: Nykaa-style Market Basket Dashboard")

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    rules = pd.read_csv("association_rules.csv")
    preds = pd.read_csv("rf_predicted_vs_actual.csv")
    segments = pd.read_csv("customer_segments.csv")
    model = joblib.load("rf_category_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return rules, preds, segments, model, label_encoder

rules, preds, segments, rf_model, label_encoder = load_data()
label_map = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))

# ========== SIDEBAR ==========
section = st.sidebar.radio("📂 Navigate", [
    "Association Rules", "Product Recommender",
    "Customer Segments", "Model Insights"])

# ========== SECTION 1: ASSOCIATION RULES ==========
if section == "Association Rules":
    st.subheader("🔗 Association Rule Explorer")
    min_lift = st.slider("Minimum Lift", 1.0, 5.0, 2.0)
    min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.5)

    rules["antecedents"] = rules["antecedents"].str.replace("frozenset", "").str.strip("(){}")
    rules["consequents"] = rules["consequents"].str.replace("frozenset", "").str.strip("(){}")

    filtered = rules[(rules["lift"] >= min_lift) & (rules["confidence"] >= min_conf)]
    st.dataframe(filtered[["antecedents", "consequents", "support", "confidence", "lift"]]
                 .sort_values(by="lift", ascending=False))

# ========== SECTION 2: PRODUCT RECOMMENDER ==========
elif section == "Product Recommender":
    st.subheader("💡 Product Recommender (Apriori-based)")

    rules["antecedents_set"] = rules["antecedents"].apply(lambda x: set(x.replace("frozenset({", "").replace("})", "").replace("'", "").split(", ")))
    rules["consequents_set"] = rules["consequents"].apply(lambda x: set(x.replace("frozenset({", "").replace("})", "").replace("'", "").split(", ")))

    product_input = st.text_input("Enter a product name (e.g., Lipstick):")

    if product_input:
        recs = rules[rules["antecedents_set"].apply(lambda x: product_input.title() in x)]
        if recs.empty:
            st.warning("❌ No recommendations found for this product.")
        else:
            st.write(recs[["antecedents", "consequents", "confidence", "lift"]]
                     .sort_values(by="lift", ascending=False).head(5))

# ========== SECTION 3: CUSTOMER SEGMENTS ==========
elif section == "Customer Segments":
    st.subheader("👥 Customer Segments (KMeans Clusters)")
    st.dataframe(segments.groupby("Segment")[["BasketSize", "TotalSpend", "Cleansing", "Hair Care", "Lips", "Make-up Eyes", "Skin"]].mean().round(2))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=segments, x="PCA1", y="PCA2", hue="Segment", palette="tab10", alpha=0.6)
    plt.title("Customer Segments (PCA Projection)")
    st.pyplot(fig)


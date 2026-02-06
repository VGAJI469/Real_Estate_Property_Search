import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Real Estate Property Search",
    page_icon="🏡",
    layout="wide"
)

# ---------------- DARK MODE ----------------
mode = st.sidebar.toggle("🌙 Dark Mode")

if mode:
    dark_css = """
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")

    df["text"] = df.apply(lambda r:
        f"{r['bedrooms']} bedroom property in {r['location']} "
        f"priced at {r['price']} with {r['area']} sqft. "
        f"{r['description']}",
        axis=1
    )

    return df

df = load_data()

st.sidebar.success(f"{len(df)} Properties Indexed")

# ---------------- FILTERS (HYBRID SEARCH) ----------------
st.sidebar.header("🔎 Filter Properties")

max_price = st.sidebar.slider(
    "Maximum Price",
    int(df.price.min()),
    int(df.price.max()),
    int(df.price.max())
)

bedroom_choice = st.sidebar.selectbox(
    "Bedrooms",
    ["Any"] + sorted(df.bedrooms.unique().tolist())
)

filtered_df = df[df["price"] <= max_price]

if bedroom_choice != "Any":
    filtered_df = filtered_df[filtered_df["bedrooms"] == bedroom_choice]

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ---------------- VECTOR DB ----------------
@st.cache_resource
def create_index(texts):
    embeddings = model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, embeddings

index, embeddings = create_index(filtered_df["text"].tolist())

# ---------------- SEARCH ----------------
def search_property(query, k=3):

    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results = filtered_df.iloc[indices[0]].copy()
    results["distance"] = distances[0]

    return results

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align: center;
background: -webkit-linear-gradient(#00c6ff, #0072ff);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;'>
🏡 AI Semantic Property Search
</h1>
""", unsafe_allow_html=True)

st.info(
"""
Try searching:

• "Affordable 2BHK near metro"  
• "Luxury 4 bedroom in Gurgaon"  
• "Family home near schools"
"""
)

# ---------------- SEARCH UI ----------------
query = st.text_input(
    "Describe your ideal property:",
    placeholder="Example: 2 bedroom apartment near metro under 80 lakh"
)

if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a search query.")

    else:

        with st.spinner("🤖 AI is searching the best properties for you..."):

            results = search_property(query)

        st.subheader("🏆 Top Matching Properties")

        if results.empty:
            st.error("No matching properties found. Try a different query.")

        for _, row in results.iterrows():

            confidence = round(1 - row["distance"], 2)

            with st.container():

                st.markdown(f"""
                ### 📍 {row['location']}

                💰 **Price:** ₹{row['price']:,}  
                🛏 **Bedrooms:** {row['bedrooms']}  
                📐 **Area:** {row['area']} sqft  

                📝 {row['description']}
                """)

                st.caption(f"AI Match Confidence: {confidence}")

                st.divider()

import matplotlib.pyplot as plt

st.subheader("📊 Market Insights: Average Property Prices")

avg_prices = df.groupby("location")["price"].mean().sort_values()

fig, ax = plt.subplots()
avg_prices.plot(kind="bar", ax=ax)

ax.set_xlabel("City")
ax.set_ylabel("Average Price")
ax.set_title("Average Apartment Prices by City")
st.info(
"Insight: Cities with higher average prices typically indicate stronger infrastructure, job markets, and housing demand."
)
st.pyplot(fig)

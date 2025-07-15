import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(layout="wide")
st.title("📚 Topic Modeling & Clustering: 20 Newsgroups")

# --------------------
# Sidebar options
# --------------------
algo = st.sidebar.radio("Select Algorithm", ["KMeans", "LDA"])
use_uploaded = st.sidebar.checkbox("Use uploaded folder", value=False)
n_topics = st.sidebar.slider("Number of Topics/Clusters", 2, 20, 5)

# --------------------
# Load documents
# --------------------
@st.cache_data
def load_docs(base_path):
    documents = []
    labels = []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                with open(fpath, 'r', encoding='latin1') as f:
                    documents.append(f.read())
                    labels.append(folder)
    return documents, labels

if use_uploaded:
    uploaded_path = st.sidebar.text_input("Enter path to folder", value=r"C:\Users\imran\OneDrive\Desktop\Celebal\20_newsgroups\20_newsgroups")
    docs, true_labels = load_docs(uploaded_path)
else:
    docs, true_labels = load_docs(r"C:\Users\imran\OneDrive\Desktop\Celebal\mini_newsgroups\mini_newsgroups")

st.write(f"📄 Loaded **{len(docs)}** documents from **{len(set(true_labels))}** categories.")

# --------------------
# Vectorization
# --------------------
if algo == "KMeans":
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2)
else:
    vectorizer = CountVectorizer(stop_words='english', max_df=0.5, min_df=2)

X = vectorizer.fit_transform(docs)
features = vectorizer.get_feature_names_out()

# --------------------
# KMeans
# --------------------
if algo == "KMeans":
    kmeans = KMeans(n_clusters=n_topics, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Truncated SVD for 2D
    svd = TruncatedSVD(n_components=2)
    X_svd = svd.fit_transform(X)

    # Plotting
    st.subheader("🔍 2D Visualization of Clusters")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_svd[:, 0], X_svd[:, 1], c=cluster_labels, cmap='tab20', s=10)
    ax.set_title("KMeans Clustering (TruncatedSVD)")
    st.pyplot(fig)

    # Export CSV
    df = pd.DataFrame({'Document': docs, 'Cluster': cluster_labels})
    csv = df.to_csv(index=False)
    st.download_button("⬇ Download Clustered Data as CSV", csv, "kmeans_clusters.csv", "text/csv")

# --------------------
# LDA
# --------------------
else:
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    st.subheader("☁️ LDA Topics as Word Clouds")
    for topic_idx, topic in enumerate(lda.components_):
        wc = WordCloud(background_color='white', width=800, height=400)
        topic_words = {features[i]: topic[i] for i in topic.argsort()[:-21:-1]}
        wc.generate_from_frequencies(topic_words)

        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(f"Topic #{topic_idx}")
        st.pyplot(fig)

    # Export topics
    st.subheader("⬇ Export LDA Topics to CSV")
    topic_data = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [features[i] for i in topic.argsort()[:-11:-1]]
        topic_data.append([f"Topic {topic_idx}"] + top_words)
    topic_df = pd.DataFrame(topic_data)
    lda_csv = topic_df.to_csv(index=False, header=False)
    st.download_button("⬇ Download LDA Topics", lda_csv, "lda_topics.csv", "text/csv")


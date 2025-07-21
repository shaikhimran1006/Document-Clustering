import streamlit as st
import joblib
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import LdaModel
from gensim import corpora
from wordcloud import WordCloud
import plotly.graph_objects as go

# Load models
kmeans_model = joblib.load("models/KMeans/kmeans_model_10_clusters.joblib")
lda_model = LdaModel.load("models/LDA/lda_model_10_topics.gensim")
dictionary = corpora.Dictionary.load("models/LDA/lda_dictionary.gensim")
tfidf_vectorizer = joblib.load("models/KMeans/tfidf_vectorizer.joblib")
nmf_model = joblib.load("models/NMF/nmf_model_10_topics.joblib")
nmf_vectorizer = joblib.load("models/NMF/nmf_vectorizer.joblib")
agglo_model = joblib.load("models/Agglomerative/agglomerative_model.pkl")
agglo_vectorizer = joblib.load("models/Agglomerative/vectorizer_agglomerative.pkl")

# Label mappings
cluster_names = {
    0: "Sports (Hockey & Baseball)",
    1: "Space & NASA",
    2: "Meta & Headers (Low Info)",
    3: "Politics & Government",
    4: "Windows & Software",
    5: "Motorcycles & Biking",
    6: "Computer Graphics",
    7: "Cryptography & Security",
    8: "Christianity & Religion",
    9: "Computer Hardware"
}
lda_topic_names = {
    0: "Christianity & Religion",
    1: "Space & NASA",
    2: "Cryptography & Security",
    3: "Middle East Politics",
    4: "Politics & Guns",
    5: "Computer Graphics & Programming",
    6: "Windows OS Support",
    7: "Automobiles & Motors",
    8: "Baseball & Sports",
    9: "Health & Medicine"
}
agglo_cluster_names = {
    0: "Space & NASA",
    1: "Computer Graphics",
    2: "Atheism & Religion",
    3: "Sports (Baseball, Hockey)",
    4: "Politics & Government",
    5: "Medical & Health",
    6: "Cars & Motorcycles",
    7: "Science & Technology",
    8: "Cryptography & Security",
    9: "Christianity & Religion"
}
nmf_topic_names = {
    0: "Cars & Engines",
    1: "Medicine & Health",
    2: "Cryptography & Security",
    3: "Politics & Religion",
    4: "Computers & Operating Systems",
    5: "Computer Graphics & Design",
    6: "Sports & Baseball",
    7: "Space & Astronomy",
    8: "Guns & Firearms",
    9: "Christian Beliefs"
}

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]

def show_wordcloud(words, title):
    wordcloud = WordCloud(background_color='black').generate(" ".join(words))
    st.image(wordcloud.to_array(), caption=title, use_container_width=True)

def donut_chart(words, weights, title):
    fig = go.Figure(data=[go.Pie(labels=words, values=weights, hole=.4)])
    fig.update_layout(title_text=title)
    st.plotly_chart(fig, use_container_width=True)

# UI
st.set_page_config(page_title="Topic Modeling Comparator", layout="wide")
st.title("📚 Topic Modeling Comparator - KMeans, LDA, Agglomerative, NMF")

user_input = st.text_area("✍️ Enter document text here:", height=200)
option = st.radio("Choose a model:", ["KMeans", "LDA", "Agglomerative", "NMF", "Compare All"], horizontal=True)

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        tokens = preprocess(user_input)
        clean_text = " ".join(tokens)

        if option in ["KMeans", "Compare All"]:
            st.subheader("🔹 KMeans Result")
            vec = tfidf_vectorizer.transform([clean_text])
            cid = kmeans_model.predict(vec)[0]
            label = cluster_names.get(cid, f"Cluster {cid}")
            st.success(f"📌 KMeans: **{label}**")
            terms = tfidf_vectorizer.get_feature_names_out()
            top_idx = kmeans_model.cluster_centers_[cid].argsort()[::-1][:10]
            top_words = [terms[i] for i in top_idx]
            weights = kmeans_model.cluster_centers_[cid][top_idx]
            st.write("🔑 Top Keywords:", ", ".join(top_words))
            col1, col2 = st.columns(2)
            with col1:
                show_wordcloud(top_words, "KMeans WordCloud")
            with col2:
                donut_chart(top_words, weights, "KMeans Top Keywords")

        if option in ["LDA", "Compare All"]:
            st.subheader("🔹 LDA Result")
            bow = dictionary.doc2bow(tokens)
            lda_topics = lda_model.get_document_topics(bow)
            if lda_topics:
                tid, _ = max(lda_topics, key=lambda x: x[1])
                label = lda_topic_names.get(tid, f"Topic {tid}")
                st.success(f"🧠 LDA: **{label}**")
                topic_words = lda_model.show_topic(tid, topn=10)
                words, weights = zip(*topic_words)
                st.write("🔑 Top Keywords:", ", ".join(words))
                col1, col2 = st.columns(2)
                with col1:
                    show_wordcloud(words, "LDA WordCloud")
                with col2:
                    donut_chart(words, weights, "LDA Top Keywords")
            else:
                st.warning("⚠️ LDA couldn't assign a topic.")

        if option in ["Agglomerative", "Compare All"]:
            st.subheader("🔹 Agglomerative Result")
            vec = agglo_vectorizer.transform([clean_text])
            cid = agglo_model.predict(vec.toarray())[0]
            label = agglo_cluster_names.get(cid, f"Cluster {cid}")
            st.success(f"📎 Agglomerative: **{label}**")
            terms = agglo_vectorizer.get_feature_names_out()
            cluster_weights = agglo_model.centroids_[cid]
            top_idx = cluster_weights.argsort()[::-1][:10]
            top_words = [terms[i] for i in top_idx]
            weights = cluster_weights[top_idx]
            st.write("🔑 Top Keywords:", ", ".join(top_words))
            col1, col2 = st.columns(2)
            with col1:
                show_wordcloud(top_words, "Agglomerative WordCloud")
            with col2:
                donut_chart(top_words, weights, "Agglomerative Top Keywords")

        if option in ["NMF", "Compare All"]:
            st.subheader("🔹 NMF Result")
            vec = nmf_vectorizer.transform([clean_text])
            topic_dist = nmf_model.transform(vec)
            tid = np.argmax(topic_dist)
            label = nmf_topic_names.get(tid, f"Topic {tid}")
            st.success(f"🔬 NMF: **{label}**")
            top_idx = nmf_model.components_[tid].argsort()[::-1][:10]
            top_words = [nmf_vectorizer.get_feature_names_out()[i] for i in top_idx]
            weights = nmf_model.components_[tid][top_idx]
            st.write("🔑 Top Keywords:", ", ".join(top_words))
            col1, col2 = st.columns(2)
            with col1:
                show_wordcloud(top_words, "NMF WordCloud")
            with col2:
                donut_chart(top_words, weights, "NMF Top Keywords")

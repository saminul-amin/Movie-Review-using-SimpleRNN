import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #667eea;
        font-size: 16px;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 20px 0;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 20px 0;
    }
    .score-box {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .header-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .example-box {
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_word_index():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index

@st.cache_resource
def load_sentiment_model():
    try:
        return load_model('simple_rnn_imdb.h5', compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Try regenerating the model with your current TensorFlow version")
        return None

word_index, reverse_word_index = load_word_index()
model = load_sentiment_model()

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text, max_words=10000):
    words = text.lower().split()
    encoded_review = []
    for word in words:
        index = word_index.get(word, 2)

        index = index + 3

        if index >= max_words:
            index = 2 
        encoded_review.append(index)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

st.markdown("""
    <div class="header-container">
        <h1 style="text-align: center; color: #667eea; margin: 0;">IMDB Movie Review Sentiment Analyzer</h1>
        <p style="text-align: center; font-size: 18px; color: #555; margin-top: 10px;">
            Discover whether your movie review is positive or negative using deep learning!
        </p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Enter Your Movie Review")
    user_input = st.text_area(
        "",
        height=200,
        placeholder="Type or paste your movie review here... For example: 'This movie was absolutely amazing! The acting was superb and the plot kept me engaged throughout.'",
        help="Enter a detailed movie review to analyze its sentiment"
    )
    
    analyze_button = st.button("Analyze Sentiment", use_container_width=True, type="primary")

with col2:
    st.markdown("### Example Reviews")
    st.markdown("""
        <div class="example-box">
            <strong>Positive:</strong><br>
            "Brilliant film with outstanding performances!"
        </div>
        <div class="example-box">
            <strong>Negative:</strong><br>
            "Waste of time. Poor acting and boring plot."
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ℹ️ How it works")
    st.info("This app uses a Recurrent Neural Network trained on 50,000 IMDB reviews to predict sentiment with high accuracy.")

if analyze_button:
    if user_input.strip():
        with st.spinner('Analyzing your review...'):

            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input, verbose=0)
            score = prediction[0][0]
            sentiment = 'Positive' if score > 0.5 else 'Negative'
            
            st.markdown("---")
            st.markdown("## Analysis Results")
            
            if sentiment == 'Positive':
                st.markdown(f"""
                    <div class="sentiment-positive">
                        😊 {sentiment.upper()} SENTIMENT 😊
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="sentiment-negative">
                        😞 {sentiment.upper()} SENTIMENT 😞
                    </div>
                """, unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Confidence Score", f"{score:.2%}")
            
            with col_b:
                st.metric("Sentiment", sentiment)
            
            with col_c:
                confidence_level = "High" if abs(score - 0.5) > 0.3 else "Medium" if abs(score - 0.5) > 0.15 else "Low"
                st.metric("Confidence Level", confidence_level)
            
            st.markdown("### Sentiment Score Visualization")
            st.progress(float(score))
            
            col_neg, col_pos = st.columns(2)
            with col_neg:
                st.markdown(f"**Negative:** {(1-score)*100:.1f}%")
            with col_pos:
                st.markdown(f"**Positive:** {score*100:.1f}%")
                
    else:
        st.warning("⚠️ Please enter a movie review to analyze.")

with st.sidebar:
    st.markdown("## Model Information")
    st.markdown("""
    - **Model Type:** Simple RNN
    - **Dataset:** IMDB Reviews
    - **Training Samples:** 50,000
    - **Vocabulary Size:** ~88,000
    - **Max Review Length:** 500
    """)
    
    st.markdown("---")
    st.markdown("## Tips for Best Results")
    st.markdown("""
    - Write detailed reviews (50+ words)
    - Use clear language
    - Express your opinion clearly
    - Include specific details
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This sentiment analyzer uses deep learning to classify movie reviews as positive or negative based on the text content.
    """)
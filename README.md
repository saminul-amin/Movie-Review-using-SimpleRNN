# IMDB Movie Review Sentiment Analyzer

A professional web application that classifies movie reviews as positive or negative using a Recurrent Neural Network (RNN) trained on the IMDB reviews dataset. Built with TensorFlow, Streamlit, and NumPy.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)

## Features

- **Real-time Sentiment Analysis**: Instantly classify movie reviews as positive or negative
- **Deep Learning Model**: Powered by a trained Simple RNN with ~88,000 vocabulary
- **Beautiful UI**: Modern gradient design with intuitive layout
- **Confidence Scoring**: See prediction confidence levels and visualizations
- **Example Reviews**: Quick examples to test the system
- **Responsive Design**: Works seamlessly on different screen sizes

## Live Link

[Try the live application here!](https://movie-sentiment-analysis-using-simplernn.streamlit.app/)

## How It Works

### 1. Text Processing Pipeline
- Converts input text to lowercase
- Maps words to indices using IMDB word index
- Pads sequences to 500 words maximum
- Handles unknown words with special tokens

### 2. Model Architecture
- **Type**: Simple Recurrent Neural Network (RNN)
- **Input**: Sequence of word indices (max length: 500)
- **Output**: Single probability score (0 = Negative, 1 = Positive)
- **Training Data**: 50,000 IMDB movie reviews
- **Vocabulary**: Approximately 88,000 unique words

### 3. Prediction Logic
- **Score > 0.5**: Positive sentiment
- **Score ≤ 0.5**: Negative sentiment
- Confidence levels: High (>80%), Medium (65-80%), Low (<65%)


## Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **NumPy**: Numerical computations
- **IMDB Dataset**: 50,000 labeled movie reviews


## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- TensorFlow team for the excellent deep learning framework
- Streamlit for the amazing web app framework
- IMDB for the movie reviews dataset
- All contributors and users of this project

---

<div align="center">
  
Made with ❤️ by Md. Saminul Amin

[![Star](https://img.shields.io/github/stars/saminul-amin/Movie-Review-using-SimpleRNN?style=social)](https://github.com/saminul-amin/Movie-Review-using-SimpleRNN)
[![Fork](https://img.shields.io/github/forks/saminul-amin/Movie-Review-using-SimpleRNN?style=social)](https://github.com/saminul-amin/Movie-Review-using-SimpleRNN)

</div>
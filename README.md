# Sentiment-Analysis-on-IMDB-reviews-using-deep-learning-and-NLP
Movie Review Sentiment Analysis using IMDB Dataset
Project Overview
This project aims to create a robust sentiment analysis model for movie reviews, providing valuable insights for movie studios, distributors, and marketers. By accurately capturing public sentiment towards films, the model enables informed decision-making within the film industry. It helps enhance marketing strategies, tailor movie content to audience preferences, and improve the overall movie-making process.

The dataset used contains 50,000 movie reviews from IMDB, widely used for sentiment analysis in Natural Language Processing (NLP). Each review is labeled as either "positive" or "negative," and the dataset is balanced with an equal representation of both sentiments, making it ideal for training machine learning models.

Objective
Analyze and predict sentiment (positive/negative) of movie reviews.
Provide actionable insights for marketing strategies, content creation, and audience engagement.
Dataset Description
The IMDB dataset contains the following columns:

review: Text of the movie review.
sentiment: Sentiment label (positive/negative).
The dataset is preprocessed to ensure:

Standardization: Treating words like "Movie" and "movie" the same.
Data cleaning: Removing irrelevant content and focusing on meaningful words for better analysis.
Tokenization: Breaking text into individual words for enhanced processing and analysis.
Stopword removal: Filtering out common words (e.g., "and," "the") using NLTK to emphasize significant terms contributing to sentiment.
Approach
1. GloVe (Global Vectors for Word Representation) Embedding
GloVe is a word embedding technique that represents words as dense vectors in a continuous vector space. These embeddings capture semantic relationships and contextual meanings based on word co-occurrence in a large corpus.

Example:

Review A: "The acting was superb, and the storyline was captivating."
Review B: "The movie was dull, and the plot was predictable."
Relevant words extracted: superb, captivating, dull, predictable.

2. Convolutional Neural Network (CNN) Architecture
Embedding Layer: Transforms words into GloVe embeddings.
Convolutional Layer: Uses 128 filters with ReLU activation to capture initial patterns in the text.
Dropout Layer: Regularization to prevent overfitting.
Second Convolutional Layer: Uses 256 filters to learn deeper patterns.
Global Max Pooling: Retains the most significant features.
Dense Layer with Sigmoid Activation: Outputs sentiment prediction (positive or negative).
CNN Results:

Epochs: 10
Batch Size: 128
Testing Accuracy: 85.03%
Training Accuracy: 96.58%
Testing Loss: 0.4019
Training Loss: 0.1025
3. Long Short-Term Memory (LSTM) Architecture
Embedding Layer: Transforms words into embeddings.
LSTM Layer: Captures sequential dependencies in text.
Dense Output Layer: Performs sentiment prediction using sigmoid activation.
Reason for Using LSTM:

Recognizes sentiment shifts (e.g., negations).
Retains long-context information.
Performs well with smaller datasets and varying sentence lengths.
Understands how word order affects meaning.
LSTM Results:

Epochs: 10
Batch Size: 128
Testing Accuracy: 85.03%
Training Accuracy: 96.58%
Testing Loss: 0.4019
Training Loss: 0.1025

Impact of Sentiment Analysis on IMDB
Guides Content Creation: Helps creators align movie content with audience preferences.
Informs Audience Choices: Empowers viewers to make informed decisions based on sentiment.
Enhances Reputation Management: Allows quick responses to viewer feedback.
Optimizes Marketing: Adjusts promotional strategies based on early audience reactions.
Improves Content Recommendations: Tailors movie suggestions based on viewer sentiment.

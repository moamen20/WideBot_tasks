# Text Classification using Support Vector Classifier (SVC) 

This repository contains the code for training and evaluating a text classification model using the Support Vector Classifier (SVC) algorithm. The model is designed to classify stories into different topics based on their titles and content.

## Feature Engineering

The first step in the training process involves feature engineering. We combine the 'story' and 'title' columns of the dataset to create a new feature called 'title_story'. This new feature concatenates the title and story text using a special delimiter '@@'. This step is performed to capture information from both the title and content of the news articles, allowing the model to leverage more context for classification.

## Data Splitting

Next, we split the data into training and test sets using a test size of 20% of the total data.

## Model Building

The core of the training process involves building the classification model. Support Vector Classifier (SVC) algorithm is used for this task. The SVC is a powerful supervised learning algorithm used for classification tasks. In our case, we set the kernel type to 'linear', which means we are using a linear kernel function. Other kernel options, such as 'poly' or 'rbf', can be explored to experiment with different decision boundaries.

To transform the text data into numerical features, we use the TfidfVectorizer. This vectorizer calculates the Term Frequency-Inverse Document Frequency (TF-IDF) values for each word in the text corpus. The resulting TF-IDF matrix represents the importance of each word in the context of each news article, and it serves as the input to the SVC classifier.

## Model Evaluation

After training the model, we proceed with evaluating its performance on the test data. The evaluation is based on various metrics, including precision, recall, F1-score, and accuracy. These metrics provide insights into the model's ability to correctly classify news articles into their respective topics.

The classification report is used to obtain precision, recall, F1-score, and support for each class. Precision measures the proportion of correctly classified instances for each class out of all instances predicted for that class. Recall (also called sensitivity) measures the proportion of correctly classified instances for each class out of all instances belonging to that class in the test set. F1-score is the harmonic mean of precision and recall, providing a balanced measure for binary classification. Support is the number of occurrences of each class in the test set.

## Hyperparameter Tuning

The training process involves hyperparameter tuning to optimize the model's performance. In this implementation, we focus on adjusting the kernel type for the SVC algorithm, the min_df and max_df parameters of the TfidfVectorizer, and the max_features parameter of the vectorizer. By experimenting with different hyperparameter settings, we can find the combination that yields the best results for our specific classification task.

## Enhancements

To further improve the model's performance, additional enhancements can be considered:

1. Text Preprocessing: Experiment with different text preprocessing techniques, such as stemming, lemmatization, or removing stopwords, to see how they impact model performance. These techniques can help reduce noise and dimensionality in the data and may lead to better generalization.

2. Domain-Specific Embeddings: If you have a large corpus of domain-specific text data, consider using pre-trained word embeddings specific to your domain. Domain-specific embeddings capture context and semantic information that is relevant to your specific topic classification task, potentially improving model accuracy.

In summary, the training process involves feature engineering, data splitting, model building, and model evaluation. By carefully tuning hyperparameters and exploring different enhancements, we can achieve a well-performing text classification model. The goal is to optimize the model's ability to learn and generalize patterns in the text data, leading to accurate topic classification for stories.

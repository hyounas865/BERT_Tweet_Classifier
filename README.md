# Tweet Emotion Detection using BERT

## Project Overview
This project utilizes the pre-trained **BERT** model to classify the sentiment or emotion of tweets. The dataset used for this project contains tweets labeled with 13 different emotions. The project involves tokenizing the tweets using BERT's tokenizer, training a neural network based on the BERT model, and using this model to predict the sentiment of new tweets.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
  - [Data Loading](#data-loading)
  - [Tokenization](#tokenization)
  - [Model Training](#model-training)
  - [Training & Validation](#training-validation)
  - [Prediction](#prediction)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Dataset
The dataset used is **tweet_emotions.csv**, which contains tweets categorized into 13 different sentiment classes. These are:

- Empty
- Sadness
- Enthusiasm
- Neutral
- Worry
- Surprise
- Love
- Fun
- Hate
- Happiness
- Boredom
- Relief
- Anger

The dataset can be found on Kaggle, and it is used to train the model by labeling each tweet with one of the above sentiments.

---

## Project Workflow

### 1. Data Loading
- The dataset is loaded into a pandas DataFrame.
- Token lengths are calculated for each tweet.
- Sentiments are encoded using integer values to be used for model training.

### 2. Tokenization
- Tweets are tokenized using `BertTokenizer` from the Hugging Face transformers library.
- Tokenized tweets are padded or truncated to a maximum length of 256 tokens.
- Attention masks are generated to distinguish between actual token data and padding.

### 3. Model Training
- The **BERT** model is loaded from the Hugging Face library with pre-trained weights.
- Input layers for `input_ids` and `attention_masks` are defined.
- An intermediate dense layer with ReLU activation is added to enhance learning.
- The output layer uses softmax activation for classification into one of 13 sentiment categories.

### 4. Training & Validation
- The dataset is shuffled and split into **training** and **validation** sets.
- The model is trained for one or more epochs, depending on user configuration.
- The trained model is saved for future predictions.

### 5. Prediction
- The saved model is loaded and used to predict the sentiment of new tweets.
- The `prepare_data` function processes the tweet text and tokenizes it.
- The `make_prediction` function takes the processed input and returns the predicted sentiment.

---

## Model Architecture
The architecture of the model consists of:

1. **Pre-trained BERT model** with frozen weights from the Hugging Face library.
2. **Intermediate dense layer** with ReLU activation for additional learning.
3. **Output softmax layer** for multi-class classification across 13 emotion categories.

---

## Requirements
- Python 3.x
- TensorFlow
- Hugging Face transformers library
- pandas
- numpy
- tqdm

---

## Installation

To install the necessary packages, use the following command:

## Results

The model outputs one of the 13 sentiment categories based on the input tweet. Evaluation metrics such as accuracy, loss, and validation accuracy are calculated during training to track model performance.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

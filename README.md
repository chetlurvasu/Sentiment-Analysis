# Sentiment-Analysis

This project aims to perform sentiment analysis on textual data using a Long Short-Term Memory (LSTM) network, a type of recurrent neural network (RNN). Sentiment analysis involves determining the sentiment or opinion expressed in a piece of text, whether it's positive or negative.

Dataset

The dataset used for this project consists of textual reviews along with their corresponding sentiment labels. The reviews are preprocessed by converting them to lowercase and removing special characters using regular expressions.

Preprocessing

Text data is tokenized and converted into sequences of integers using the Keras Tokenizer. Padding is applied to ensure uniform sequence length required by the LSTM model.

Model Architecture

The LSTM model consists of an embedding layer, an LSTM layer, and a dense output layer with softmax activation. The model is compiled with categorical cross-entropy loss and the Adam optimizer.

Training

The model is trained on the training data split from the dataset. Training is performed for a specified number of epochs with a defined batch size.

Evaluation

The trained model is evaluated on the test data to measure its performance in terms of accuracy. The accuracy metric indicates the percentage of correctly classified sentiments.

Prediction

The trained model can be used to predict the sentiment of new textual data. The input text is preprocessed and passed through the model, which outputs the predicted sentiment (positive or negative).

Dependencies
1. Python 3.x
2. TensorFlow
3. Keras
4. numpy
5. pandas
6. scikit-learn

Usage
1. Clone the repository.
2. Install the dependencies listed in requirements.txt.
3. Prepare your dataset or use the provided one.
4. Train the model using train.py.
5. Evaluate the model using evaluate.py.
6. Make predictions using predict.py.
   
Feel free to contribute or provide feedback!

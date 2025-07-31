"""
Implementation of Sentiment Analysis of IDMb Movie Reviews with a Custom Neural Network
Developed an end-to-end sentiment analysis solution for classifying IMDb movie reviews as positive or negative
This project's core involved implementing a Multi-Layer Perceptron (MLP) Neural Network entirely from scratch in Python
The pipeline integrated robust natural language processing (NLP) for text preprocessing, encompassing cleaning and advanced linguistic analysis using spaCy for efficient tokenization, lemmatization, and stop word removal
Processed text was then transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency), preparing the data for the custom-built model
Model training involved optimizing the custom MLP using mini-batch gradient descent, with performance monitored across training, validation, and test datasets
"""
import os
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path
import tarfile
import nltk # Natural Language Tool Kit library
import spacy # Natural Language Tool Kit library (has faster dictionary look ups than nltk, and less computationally intensive)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool, cpu_count
'''
Note: Even with spaCy, for 50,000 documents, it will still take some time as it's doing complex linguistic analysis. But its usually faster than NLTK at this scale.
'''

def load_imdb_data(path, extract_path):
  texts = []
  labels = []

  data_dir = Path(extract_path)

  if not data_dir.exists():
    try:
      with tarfile.open(path, "r:gz") as tar:
        tar.extractall(path=str(Path("."))) # extracting tar file to current working dir
    except Exception as e:
        print(f"cannot load file in path {path} due to {e}")
        return texts, labels # None, None

  for split in ["train", "test"]:
    for sentiment in ["pos", "neg"]:
      folder_path = os.path.join(data_dir, split, sentiment)
      if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
          if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
              texts.append(f.read()) # appending the text as it is
              labels.append(1 if sentiment == "pos" else 0) # appending binary classes 1, 0 for 'pos' and 'neg' sentiment respectively
      else:
        print(f"folder with path '{folder_path}' does not exist")

  return texts, labels

def clean_text(text):
  text = text.lower() # lowercasing text
  text = re.sub(r'<br />', ' ', text) # removing HTML break '<br />' tags
  text = re.sub(r'[^a-z\s]', '', text) # removing punctuation and numbers to keep only letters and spaces
  return text

def tokenize_and_lemmantize_text(text):
  lemma_tokens = []
  lemmatizer = WordNetLemmatizer() #unfortunately the WordNetLemmatizer
  tokens = nltk.word_tokenize(text)

  for word in tokens:
    if word not in stopwords.words('english') and word.isalpha(): # Checking if it's not a stop word and is primarily alphabetic
      lemma = lemmatizer.lemmatize(word) #reduces inflected forms of a word to its base or dictionary form (lemma) using wordnet's morphological analysis and vocabulary
      lemma_tokens.append(lemma)

  return " ".join(lemma_tokens) # return as string for TF-IDF vectorization

class NeuralNetwork:
  def __init__(self, layer_sizes, learning_rate):
    self.layer_sizes = layer_sizes
    self.learning_rate = learning_rate

    self.list_of_weight_matrices = [] #list of weight matrices between the layers
    self.list_of_bias_vectors = [] #list of bias vectors for each layer

    for l in range(len(self.layer_sizes) - 1):
      # For hidden layers (using ReLU), use He initialization
      # For the output layer (using Sigmoid), Xavier is more appropriate, but He can also work reasonably well.
      # A common practice is to use He for all layers if ReLU is prevalent, or
      # be more precise and use He for ReLU layers and Xavier for Sigmoid/Tanh.

      # He Initialization for layers leading to ReLU activations
      if l < len(self.layer_sizes) - 2: # All hidden layers
          limit = np.sqrt(6 / self.layer_sizes[l])
      else: # Output layer (leading to Sigmoid)
          # Xavier Initialization (Glorot Uniform) for the output layer
          limit = np.sqrt(6 / (self.layer_sizes[l] + self.layer_sizes[l+1]))

      weight_matrix = np.random.uniform(-limit, limit, (self.layer_sizes[l], self.layer_sizes[l+1]))
      self.list_of_weight_matrices.append(weight_matrix)

    for l in range(len(self.layer_sizes) - 1): # except for the input layer, each layer will have a bias vector
      bias_vector = np.zeros((1, self.layer_sizes[l+1]))
      self.list_of_bias_vectors.append(bias_vector)

  def get_parameters(self):
    return self.list_of_weight_matrices, self.list_of_bias_vectors

  #Activation Functions and their Derivatives
  def sigmoid(self, z): #Sigmoid is used as activations for output layer alone
    return 1 / (1 + np.exp(-np.clip(z, -500, 500))) # clipping values to prevent overflow for np.exp() with very large/small numbers

  def sigmoid_derivative(self, a):
      return a * (1 - a)

  def relu(self, z): #ReLU is used as activations for hidden layers
      return np.maximum(0, z)

  def relu_derivative(self, a):
      return (a > 0).astype(float) # The derivative is 1 for positive values, 0 for negative/zero values

  # Loss function
  def binary_cross_entropy(self, y_true, y_pred, eps=1e-10):
      y_pred = np.clip(y_pred, eps, 1 - eps) # Clip predictions to prevent log(0) or log(1) issues
      return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

  # Derivative of Loss function w.r.t. predictions a_out aka y_pred
  def binary_cross_entropy_derivative(self, y_true, y_pred, eps=1e-10):
      y_pred = np.clip(y_pred, eps, 1 - eps)
      return (y_pred - y_true) / (y_pred * (1 - y_pred))
  #                                                            0 1 2 3
  def forward_propagation(self, X):  #E.g. self.layer_sizes = [8,5,4,1]
    self.z_values = [] # z_values stores the pre-activation (intermediate) outputs from each layer
    self.a_values = [X] # a_values stores the activation outputs from each layer
    # NOTE: There are actually no activations used in the input layer, but inorder to maintain consistency (i.e. to avoid a_values[0] to be None),
    # we consider the activation outputs of the input layer (a_values[0]) as the input (X)
    a = X
    for l, (W, b) in enumerate(zip(self.list_of_weight_matrices, self.list_of_bias_vectors)): # when l == 0, that means we are in the first hidden layer, not the input layer
      z = a @ W + b
      if l < len(self.layer_sizes) - 2:
        a = self.relu(z) #use ReLU for hidden layers
      else:
        a = self.sigmoid(z) #use sigmoid for output layer
      self.z_values.append(z)
      self.a_values.append(a)

    return a #the final output of network

  def backward_propagation(self, X, y):
    loss = self.binary_cross_entropy(y, self.a_values[-1])

    #--------------------Chain Rule part-------------------------------
    # computing gradient for output layer alone
    dloss_da_out = self.binary_cross_entropy_derivative(y, self.a_values[-1])
    da_out_dz_out = self.sigmoid_derivative(self.a_values[-1])
    dloss_dz_out = dloss_da_out * da_out_dz_out

    self.dloss_dW = [None] * (len(self.layer_sizes) - 1) # initializing a list of weight gradients of size (len(self.layer_sizes) - 1)
    self.dloss_db = [None] * (len(self.layer_sizes) - 1) # initializing a list of bias gradients of size (len(self.layer_sizes) - 1)

    for l in reversed(range(len(self.layer_sizes) - 1)): #start computing gradients from the second-last layer (hidden layer right before output layer)
        a_prev = self.a_values[l]
        self.dloss_dW[l] = a_prev.T @ dloss_dz_out
        self.dloss_db[l] = np.sum(dloss_dz_out, axis=0, keepdims=True)

        if l != 0:
          # Don't compute the gradient of the loss with respect to the input layer activations (dloss_dz_out for input layer),
          # because input neurons don’t have weights or biases to update, they are just the raw inputs (X).
          dloss_da = dloss_dz_out @ self.list_of_weight_matrices[l].T
          da_dz = self.relu_derivative(self.a_values[l])
          dloss_dz_out = dloss_da * da_dz
    #------------------------------------------------------------------

    #---------------------Gradient Descent part------------------------
    # Update weights and biases using gradient descent
    for l in range(len(self.layer_sizes) - 1):
        self.list_of_weight_matrices[l] -= self.learning_rate * self.dloss_dW[l]
        self.list_of_bias_vectors[l] -= self.learning_rate * self.dloss_db[l]
    #------------------------------------------------------------------

    # for evaluation at each pass
    return loss

  def total_gradient_norm(self):
    total_norm = 0
    for dw, db in zip(self.dloss_dW, self.dloss_db):
        total_norm += np.linalg.norm(dw)**2 + np.linalg.norm(db)**2
    return np.sqrt(total_norm)

  def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
      no_of_samples = X_train.shape[0]

      for epoch in range(epochs):
          start_time = time.time()

          # Shuffle training data at epoch start
          p = np.random.permutation(no_of_samples)
          X_train_shuffled = X_train[p]
          y_train_shuffled = y_train[p]

          # Process mini-batches
          for start in range(0, no_of_samples, batch_size):
              end = start + batch_size
              X_batch = X_train_shuffled[start:end]
              y_batch = y_train_shuffled[start:end]

              self.forward_propagation(X_batch)
              self.backward_propagation(X_batch, y_batch)

          # After all batches, evaluate on full training set
          train_output = self.forward_propagation(X_train)

          train_loss = self.binary_cross_entropy(y_train, train_output)
          train_pred = (train_output > 0.5).astype(int)
          train_accuracy = accuracy_score(y_train, train_pred)
          train_precision = precision_score(y_train, train_pred, zero_division=1)
          train_recall = recall_score(y_train, train_pred)
          train_f1 = f1_score(y_train, train_pred, zero_division=1)

          # evaluate on validation set
          val_output = self.forward_propagation(X_val)

          val_loss = self.binary_cross_entropy(y_val, val_output)
          val_pred = (val_output > 0.5).astype(int)
          val_accuracy = accuracy_score(y_val, val_pred)
          val_precision = precision_score(y_val, val_pred, zero_division=1)
          val_recall = recall_score(y_val, val_pred)
          val_f1 = f1_score(y_val, val_pred, zero_division=1)

          end_time = time.time()

          if epoch % 10 == 0:
              print(f"Epoch {epoch}\t({end_time - start_time:.2f}s):")
              print(f"  Train Loss: {train_loss:.4f}   Val Loss: {val_loss:.4f}")
              print(f"  Train Accuracy: {train_accuracy:.4f}   Val Accuracy: {val_accuracy:.4f}")
              print(f"  Train Precision: {train_precision:.4f}   Val Precision: {val_precision:.4f}")
              print(f"  Train Recall: {train_recall:.4f}   Val Recall: {val_recall:.4f}")
              print(f"  Train F1 Score: {train_f1:.4f}   Val F1: {val_f1:.4f}")
              print(f"  Total Gradient Norm: {self.total_gradient_norm():.4f}")

  def predict(self, X_test):
    test_pred = self.forward_propagation(X_test)
    return test_pred


if __name__ == "__main__":
  #nltk.download('stopwords') # list of extremely common English words that don't carry much meaning or sentiment and are ommitted while preprocessing th dataset. Eg. "a", "an", "you", etc
  #nltk.download('wordnet') # a large lexical database of English, organized by semantic relations between words, rather than just alphabetically like a dictionary
  #nltk.download('omw-1.4') # an extension of WordNet that aims to provide lexical databases for multiple languages
  #nltk.download('punkt')

  # Loading IMDB dataset
  texts, labels = load_imdb_data(path="aclImdb_v1.tar.gz", extract_path="aclImdb")
  df = pd.DataFrame({'text': texts, 'sentiment': labels})
  #print(df.info(), df.head(), df.tail())
  #print(df.shape) #(50000, 2)

  # Now, preprocessing the texts from dataset
  print("Preprocessing texts...")
  #df['processed_text'] = df['text'].apply(clean_text)
  #df['processed_text'] = df['processed_text'].apply(tokenize_and_lemmantize_text)

  # With spaCy based processing (much faster than NLTK)
  # You can still use multiprocessing with spaCy, but spaCy's nlp.pipe is also very efficient and designed for large datasets.
  processed_texts = []
  nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner']) # loads Tok2Vec, Tagger (POS tagging), Attribute Ruler, Lemmatizer, and disabling the Parser and named Entity Recognition as part of the spacy model pipeline

  for doc in nlp.pipe(df['text'].apply(clean_text), batch_size=1000, n_process=cpu_count()): # The 50000 texts are processed in batches of 1000 by each of the worker processes (equal to number of CPU cores) parallely. Each worker process executes the pipeline (Tok2Vec -> Tagger (POS tagging) -> Attribute Ruler -> Lemmatizer) for those 1000 each, and returns the combined results (combined tokens) in a 'doc'.
      lemma_tokens = []
      for token in doc:
          if not token.is_stop and token.is_alpha: # tokens that are stop words and non-alphabetic are not considered
              lemma_tokens.append(token.lemma_) #.lemma_ returns the lemma of that token
      processed_texts.append(" ".join(lemma_tokens))
  df['processed_text'] = processed_texts
  print("Preprocessing texts completed.")

  #print(df.head())

  # Now, splitting the dataset into 70% train set, 15% validation set and 15% test set
  # splitting the dataset into training+validation set and test set (85% train+val and 15% test)
  X_temp, X_test, y_temp, y_test = train_test_split(
      df['processed_text'], df['sentiment'], test_size=0.15, random_state=42, stratify=df['sentiment']
  )
  X_test_duplicate = X_test
  # splitting the 85% of train+validation set into 15% for validation set and 15% for test set
  X_train, X_val, y_train, y_val = train_test_split(
      X_temp, y_temp, test_size=0.17, random_state=42, stratify=y_temp
  ) # (15% / 85%) * 100 = 17%

  # performing TD_IDF vectorization on the splits (NOTE: TD_IDF is NOT the same as WordToVec (and GloVe))
  vectorizer = TfidfVectorizer(max_features=10000)
  #The vectorizer (e.g., CountVectorizer or TfidfVectorizer) is a feature extraction tool that learns a vocabulary from your input text data (fit),
  #and then transforms your documents into a numerical matrix of token counts or TF‑IDF weights (transform or fit_transform) using TF-IDF algorithm
  # NOTE: max_feature controls the maximum number of features (tokens/words) that will be considered when building the vocabulary.
  '''
    NOTE: TD-IDF is a numerical statistic used in information retrieval and text mining to reflect the importance of a word to a document in a collection or corpus

    When TfidfVectorizer processes a corpus of text, it identifies all unique words or n-grams.
    It then calculates the term frequency (how often each word appears) across the entire corpus.
    If max_features is specified, the vectorizer will select only the top max_features words based on their term frequency, effectively limiting the size of the vocabulary.
    max_feature can be adjusted based on the size of our training set (X_train) so that the model can capture a broad no.of highly frequent unique words or n-grams to avoid underfitting.
  '''
  print("Transforming texts...")
  #X_train_vec = np.array(vectorizer.fit_transform(X_train)) # TfidfVectorizer takes your raw training data (X_train), learns the necessary features from it, transforms it into a numerical representation (like a document-term matrix), and then converts that representation into a dense NumPy array
  #X_val_vec = np.array(vectorizer.transform(X_val))
  #X_test_vec = np.array(vectorizer.transform(X_test))

  X_train_vec = vectorizer.fit_transform(X_train).toarray() # TfidfVectorizer takes your raw training data (X_train), learns the necessary features from it, transforms it into a numerical representation (like a document-term matrix), and then converts that representation into a dense NumPy array
  X_val_vec = vectorizer.transform(X_val).toarray()
  X_test_vec = vectorizer.transform(X_test).toarray()
  #This ensures that X_train_vec, X_val_vec, and X_test_vec are always 2-dimensional NumPy arrays, even if they theoretically have only one column, allowing shape[1] to be accessed correctly.
  print("Transforming texts completed.")

  y_train = np.array(y_train).reshape(-1, 1)
  y_val = np.array(y_val).reshape(-1, 1)
  y_test = np.array(y_test).reshape(-1, 1)

  # Initializing MLP for sentiment prediction
  n_inputs = X_train_vec.shape[1] #10000
  nn = NeuralNetwork(layer_sizes=[n_inputs, 256, 128, 1], learning_rate=0.001)

  print("Model training initiated...")
  nn.train(X_train_vec, y_train, X_val_vec, y_val, epochs=70, batch_size=64)
  print("Model training completed.")

  #list_of_weight_matrices, list_of_bias_vectors = nn.get_parameters()
  #for l in range(len(layer_sizes) - 1):
  #  print(f"Learned weights between layer #{l} to layer #{l+1}: \n", list_of_weight_matrices[l])
  #  print(f"Learned biases for layer #{l+1}: \n", list_of_bias_vectors[l])

  print("Predictions:\n")
  for i in range(len(X_test_vec)):
      y_pred = nn.predict(X_test_vec[i].reshape(1, -1))
      print(f"Review: {X_test_duplicate.iloc[i]}, Actual Sentiment: {'positive' if y_test[i][0] == 1 else 'negative'}, Predicted Sentiment: {'positive' if y_pred > 0.5 else 'negative'}")

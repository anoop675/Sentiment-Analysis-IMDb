# Sentiment Analysis on IMDb Movie Review using custom Multilayer Perceptron and TF-IDF vectorization
This project demonstrates the end-to-end development of a sentiment analysis model for movie reviews, from raw text preprocessing to a custom-built deep learning model. The core innovation lies in implementing a Multi-Layer Perceptron (MLP) Neural Network entirely from scratch in Python, showcasing a deep understanding of neural network fundamentals.

Data Acquisition & Preparation:
  Developed a robust function to load and structure a large dataset of 50,000 IMDb movie reviews.
  Implemented comprehensive text preprocessing, including lowercasing, removal of HTML tags (<br />), and elimination of punctuation and numbers, focusing solely    on alphabetic characters.
  Utilized spaCy for efficient tokenization, lemmatization (reducing words to their base forms), and stop word removal, leveraging its performance advantages over   NLTK for large datasets.

Feature Engineering (TF-IDF Vectorization):
  Transformed the cleaned and lemmatized text data into numerical representations using TF-IDF (Term Frequency-Inverse Document Frequency).
  Configured TfidfVectorizer to select the top 10,000 most relevant features, effectively managing dimensionality while capturing significant word importance.

Custom Neural Network (Multi-Layer Perceptron) Implementation:
  Designed and built a feed-forward neural network from scratch in Python, without relying on high-level deep learning frameworks like TensorFlow or PyTorch.
  Architecture: Implemented a multi-layered network with customizable layer sizes, including input, hidden, and output layers.
  Weight Initialization: Employed He Initialization for ReLU-activated hidden layers and Xavier (Glorot Uniform) Initialization for the Sigmoid-activated output     layer to ensure stable training and mitigate vanishing/exploding gradients.

Activation Functions: 
  Implemented ReLU for hidden layers and Sigmoid for the binary classification output layer.

Loss Function: 
  Used the Binary Cross-Entropy loss function for optimizing binary classification tasks.

Backpropagation Algorithm: 
  Implemented the backpropagation algorithm using the chain rule to calculate gradients for all weights and biases, and updated parameters using Mini-Batch          Gradient Descent.

Model Training & Evaluation:
  Split the dataset into training (70%), validation (15%), and test (15%) sets, ensuring stratified sampling to maintain class balance.
  Trained the model using mini-batches over 70 epochs, tracking performance metrics after each epoch.
  Monitored key metrics including Loss, Accuracy, Precision, Recall, and F1-Score on both training and validation sets to identify learning trends and potential     overfitting.
  Conducted final evaluation on the unseen test set using accuracy, precision, recall, F1-score, and a confusion matrix to assess generalization capability.

NOTE: To fix: Identified overfitting tendencies during training (validation loss increasing while training loss decreases), 
fix: 
    - Early stopping (TODO: make the model return learnt weights after training, to use that for prediction)
    - L1/L2 regularization.

Technologies Used: Python, NumPy, Pandas, SpaCy, NLTK, Scikit-learn (for TF-IDF, train/test split, and evaluation metrics).

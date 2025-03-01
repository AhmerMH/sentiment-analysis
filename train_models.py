import pandas as pd
from sklearn.model_selection import train_test_split
from app.models.classical_models import ClassicalModels
from app.models.deep_learning_models import DeepLearningModels

def train():
    # Load dataset (example using IMDb or Twitter sentiment dataset)
    # For example:
    # df = pd.read_csv('sentiment_dataset.csv')
    # X = df['text'].values
    # y = df['sentiment'].values  # Binary: 0=negative, 1=positive
    
    # For this example, let's assume we have a dataset in the right format:
    # Download some dataset like IMDb or Twitter sentiment analysis dataset
    
    print("Loading dataset...")
    # This is a placeholder, you'll need to replace with actual data loading
    # Example with IMDb from keras:
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Load IMDb dataset
    max_features = 10000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    
    # Convert indices back to words
    word_index = imdb.get_word_index()
    reverse_word_index = {i+3: word for word, i in word_index.items()}
    reverse_word_index[0] = '<PAD>'
    reverse_word_index[1] = '<START>'
    reverse_word_index[2] = '<UNK>'
    
    # Convert sequences of indices to text
    X_train_text = [' '.join([reverse_word_index.get(i, '?') for i in sequence]) for sequence in x_train]
    X_test_text = [' '.join([reverse_word_index.get(i, '?') for i in sequence]) for sequence in x_test]
    
    print("Training classical models...")
    classical_models = ClassicalModels()
    classical_models.train(X_train_text, y_train)
    
    print("Training deep learning models...")
    dl_models = DeepLearningModels()
    dl_models.train_tf_model(X_train_text, y_train, epochs=2)  # Fewer epochs for demonstration
    dl_models.train_pt_model(X_train_text, y_train, epochs=2)  # Fewer epochs for demonstration
    
    print("Training complete!")

if __name__ == "__main__":
    train()

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import numpy as np
import os
import pickle

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        return out

class DeepLearningModels:
    def __init__(self):
        self.max_words = 10000
        self.max_sequence_length = 200
        self.embedding_dim = 100
        self.tf_tokenizer = Tokenizer(num_words=self.max_words)
        self.pt_tokenizer = Tokenizer(num_words=self.max_words)
        
        # TensorFlow model
        self.tf_model = None
        
        # PyTorch model
        self.pt_model = None
        
        # Check if models are trained and saved
        self.tf_trained = os.path.exists('app/models/saved/tf_lstm_model.h5')
        self.pt_trained = os.path.exists('app/models/saved/pt_lstm_model.pt')
        
        if self.tf_trained:
            self.tf_model = load_model('app/models/saved/tf_lstm_model.h5')
            with open('app/models/saved/tf_tokenizer.pickle', 'rb') as handle:
                self.tf_tokenizer = pickle.load(handle)
        
        if self.pt_trained:
            # PyTorch model needs to be initialized before loading weights
            self.pt_model = SimpleRNN(self.max_words, self.embedding_dim, 128)
            self.pt_model.load_state_dict(torch.load('app/models/saved/pt_lstm_model.pt'))
            self.pt_model.eval()
            with open('app/models/saved/pt_tokenizer.pickle', 'rb') as handle:
                self.pt_tokenizer = pickle.load(handle)
    
    def build_tf_model(self):
        """Build TensorFlow LSTM model"""
        model = Sequential()
        model.add(Embedding(self.max_words, self.embedding_dim, input_length=self.max_sequence_length))
        model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def train_tf_model(self, X_train, y_train, epochs=3, batch_size=64):
        """Train TensorFlow model"""
        self.tf_tokenizer.fit_on_texts(X_train)
        sequences = self.tf_tokenizer.texts_to_sequences(X_train)
        X_data = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        self.tf_model = self.build_tf_model()
        self.tf_model.fit(X_data, np.array(y_train), epochs=epochs, batch_size=batch_size, validation_split=0.2)
        
        # Save model and tokenizer
        os.makedirs('app/models/saved', exist_ok=True)
        self.tf_model.save('app/models/saved/tf_lstm_model.h5')
        with open('app/models/saved/tf_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tf_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.tf_trained = True
    
    def train_pt_model(self, X_train, y_train, epochs=3, batch_size=64):
        """Train PyTorch model"""
        self.pt_tokenizer.fit_on_texts(X_train)
        sequences = self.pt_tokenizer.texts_to_sequences(X_train)
        X_data = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        # Initialize model
        self.pt_model = SimpleRNN(self.max_words, self.embedding_dim, 128)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X_data, dtype=torch.long)
        y_tensor = torch.tensor(y_train, dtype=torch.float).view(-1, 1)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.pt_model.parameters())
        
        # Training loop
        for epoch in range(epochs):
            for i in range(0, len(X_data), batch_size):
                optimizer.zero_grad()
                
                # Get batch
                X_batch = X_tensor[i:i+batch_size]
                y_batch = y_tensor[i:i+batch_size]
                
                # Forward pass
                outputs = self.pt_model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
        
        # Save model and tokenizer
        os.makedirs('app/models/saved', exist_ok=True)
        torch.save(self.pt_model.state_dict(), 'app/models/saved/pt_lstm_model.pt')
        with open('app/models/saved/pt_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.pt_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.pt_model.eval()
        self.pt_trained = True
    
    def predict(self, text):
        """Make predictions with deep learning models"""
        results = {}
        
        # TensorFlow prediction
        if self.tf_trained and self.tf_model:
            sequence = self.tf_tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequence, maxlen=self.max_sequence_length)
            results['tensorflow_lstm'] = float(self.tf_model.predict(padded)[0][0])
        
        # PyTorch prediction
        if self.pt_trained and self.pt_model:
            sequence = self.pt_tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequence, maxlen=self.max_sequence_length)
            with torch.no_grad():
                tensor_input = torch.tensor(padded, dtype=torch.long)
                prediction = self.pt_model(tensor_input)
                results['pytorch_lstm'] = float(prediction.item())
        
        return results

def get_sentiment_breakdown(self, text):
    """Provide detailed breakdown of sentiment classification for deep learning models"""
    
    breakdowns = {}
    
    # TensorFlow prediction
    if self.tf_trained and self.tf_model:
        sequence = self.tf_tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_sequence_length)
        
        # Get sentiment score
        score = float(self.tf_model.predict(padded)[0][0])
        
        # Create breakdown
        tf_breakdown = {
            "sentiment": "Positive" if score > 0.6 else "Neutral" if score > 0.4 else "Negative",
            "score": score,
            "model_type": "Recurrent Neural Network (LSTM)",
            "explanation": "LSTM networks analyze text sequentially, maintaining context through their memory cells."
        }
        
        # Get important words
        # This is simplified as proper attention visualization requires model modifications
        words = text.split()
        if len(words) > 10:
            # This is a simplified approach - ideally we'd use attention weights
            tf_breakdown["important_segments"] = [
                {"segment": " ".join(words[i:i+3]), "position": f"Position {i}-{i+2}"}
                for i in range(0, min(len(words), 15), 3)
            ]
        else:
            tf_breakdown["important_segments"] = [
                {"segment": word, "position": f"Position {i}"}
                for i, word in enumerate(words)
            ]
        
        breakdowns["tensorflow_lstm"] = tf_breakdown
    
    # PyTorch prediction
    if self.pt_trained and self.pt_model:
        sequence = self.pt_tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_sequence_length)
        
        with torch.no_grad():
            tensor_input = torch.tensor(padded, dtype=torch.long)
            score = float(self.pt_model(tensor_input).item())
        
        # Create breakdown
        pt_breakdown = {
            "sentiment": "Positive" if score > 0.6 else "Neutral" if score > 0.4 else "Negative",
            "score": score,
            "model_type": "PyTorch LSTM",
            "explanation": "This LSTM network processes words sequentially to capture sentiment patterns."
        }
        
        # Similar simplified important words approach
        words = text.split()
        if len(words) > 10:
            pt_breakdown["important_segments"] = [
                {"segment": " ".join(words[i:i+3]), "position": f"Position {i}-{i+2}"}
                for i in range(0, min(len(words), 15), 3)
            ]
        else:
            pt_breakdown["important_segments"] = [
                {"segment": word, "position": f"Position {i}"}
                for i, word in enumerate(words)
            ]
        
        breakdowns["pytorch_lstm"] = pt_breakdown
    
    return breakdowns

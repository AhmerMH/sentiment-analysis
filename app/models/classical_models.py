from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np

class ClassicalModels:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100)
        }
        
        # Check if models are trained and saved
        self.trained = all(os.path.exists(f'app/models/saved/{name}_model.pkl') 
                          for name in self.models.keys())
        
        if self.trained:
            self.load_models()
    
    def train(self, X_train, y_train):
        """Train all classical models"""
        # Vectorize text data
        X_vectors = self.vectorizer.fit_transform(X_train)
        
        # Train each model
        for name, model in self.models.items():
            model.fit(X_vectors, y_train)
            
        # Save models
        os.makedirs('app/models/saved', exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, f'app/models/saved/{name}_model.pkl')
        joblib.dump(self.vectorizer, 'app/models/saved/vectorizer.pkl')
        
        self.trained = True
    
    def load_models(self):
        """Load trained models from disk"""
        for name in self.models.keys():
            self.models[name] = joblib.load(f'app/models/saved/{name}_model.pkl')
        self.vectorizer = joblib.load('app/models/saved/vectorizer.pkl')
    
    def predict(self, text):
        """Make predictions with all models"""
        if not self.trained:
            raise ValueError("Models need to be trained before prediction")
        
        # Vectorize input text
        X_vector = self.vectorizer.transform([text])
        
        results = {}
        for name, model in self.models.items():
            # Get probability of positive sentiment (assuming binary classification)
            proba = model.predict_proba(X_vector)[0]
            # Some models might have different order of classes
            pos_idx = 1 if len(proba) > 1 else 0
            results[name] = float(proba[pos_idx])
            
        return results

    def get_sentiment_breakdown(self, text):
      """Provide detailed breakdown of sentiment classification for classical models"""
      
      if not self.trained:
          return {"error": "Models need to be trained before analysis"}
      
      # Vectorize input text
      X_vector = self.vectorizer.transform([text])
      
      # Get word features
      feature_names = self.vectorizer.get_feature_names_out()
      
      # Store breakdowns for each model
      breakdowns = {}
      
      for name, model in self.models.items():
          breakdown = {}
          
          # Get sentiment score
          proba = model.predict_proba(X_vector)[0]
          pos_idx = 1 if len(proba) > 1 else 0
          sentiment_score = float(proba[pos_idx])
          
          # Set sentiment category
          if sentiment_score < 0.4:
              sentiment = "Negative"
          elif sentiment_score < 0.6:
              sentiment = "Neutral" 
          else:
              sentiment = "Positive"
              
          breakdown["sentiment"] = sentiment
          breakdown["score"] = sentiment_score
          
          # Get feature importance based on model type
          if name == 'naive_bayes':
              # For Naive Bayes, we can use log probabilities
              feature_importance = model.feature_log_prob_[1] - model.feature_log_prob_[0]
              
          elif name == 'logistic_regression':
              # For logistic regression, we can use coefficients
              feature_importance = model.coef_[0]
              
          elif name == 'random_forest':
              # For random forest, we use feature importances
              feature_importance = model.feature_importances_
          
          # Get non-zero features from the vectorized text
          non_zero_indices = X_vector.nonzero()[1]
          text_features = [(feature_names[i], X_vector[0, i], feature_importance[i]) 
                          for i in non_zero_indices]
          
          # Sort by importance (absolute value)
          text_features.sort(key=lambda x: abs(x[2]), reverse=True)
          
          # Take top 10 most important features
          top_features = text_features[:10]
          
          # Format output
          breakdown["top_features"] = [
              {
                  "word": word,
                  "importance": float(importance),
                  "contribution": "Positive" if importance > 0 else "Negative"
              }
              for word, count, importance in top_features
          ]
          
          breakdowns[name] = breakdown
      
      return breakdowns



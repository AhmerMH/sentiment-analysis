from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class TransformerModels:
    def __init__(self):
        # We'll use pre-trained models for sentiment analysis
        self.models = {
            'bert': {
                'name': 'bert-base-uncased-finetuned-sst-2-english',
                'tokenizer': None,
                'model': None
            },
            'roberta': {
                'name': 'cardiffnlp/twitter-roberta-base-sentiment',
                'tokenizer': None,
                'model': None
            }
        }
        
        # Initialize all models (can be memory-intensive)
        self.load_models()
    
    def load_models(self):
        """Load pre-trained transformer models"""
        for model_key, model_info in self.models.items():
            try:
                model_info['tokenizer'] = AutoTokenizer.from_pretrained(model_info['name'])
                model_info['model'] = AutoModelForSequenceClassification.from_pretrained(model_info['name'])
            except Exception as e:
                print(f"Error loading {model_key}: {str(e)}")
    
    def predict(self, text):
        """Make predictions with transformer models"""
        results = {}
        
        for model_key, model_info in self.models.items():
            if model_info['tokenizer'] is None or model_info['model'] is None:
                continue
                
            # Tokenize and prepare input
            inputs = model_info['tokenizer'](text, return_tensors="pt", truncation=True, max_length=512)
            
            # Make prediction
            with torch.no_grad():
                outputs = model_info['model'](**inputs)
                
            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # For models like BERT finetuned on SST-2, the positive sentiment is at index 1
            positive_score = float(probabilities[0, 1].item())
            results[model_key] = positive_score
            
        return results

def get_sentiment_breakdown(self, text):
    """Provide detailed breakdown of sentiment classification for transformer models"""
    
    breakdowns = {}
    
    for model_key, model_info in self.models.items():
        if model_info['tokenizer'] is None or model_info['model'] is None:
            continue
            
        # Tokenize and prepare input
        inputs = model_info['tokenizer'](text, return_tensors="pt", truncation=True, max_length=512)
        
        # Make prediction
        with torch.no_grad():
            outputs = model_info['model'](**inputs)
            
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # For models like BERT finetuned on SST-2, the positive sentiment is at index 1
        if model_key == 'bert':
            negative_score = float(probabilities[0, 0].item())
            positive_score = float(probabilities[0, 1].item())
            scores = {"negative": negative_score, "positive": positive_score}
            sentiment = "Positive" if positive_score > 0.6 else "Neutral" if positive_score > 0.4 else "Negative"
            main_score = positive_score
        elif model_key == 'roberta':
            # For Twitter RoBERTa, often the labels are [negative, neutral, positive]
            if probabilities.shape[1] == 3:
                negative_score = float(probabilities[0, 0].item())
                neutral_score = float(probabilities[0, 1].item())
                positive_score = float(probabilities[0, 2].item())
                scores = {"negative": negative_score, "neutral": neutral_score, "positive": positive_score}
                main_score = positive_score
                # Determine sentiment from highest score
                if positive_score > negative_score and positive_score > neutral_score:
                    sentiment = "Positive"
                elif negative_score > positive_score and negative_score > neutral_score:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
            else:
                # Binary classification
                negative_score = float(probabilities[0, 0].item())
                positive_score = float(probabilities[0, 1].item())
                scores = {"negative": negative_score, "positive": positive_score}
                sentiment = "Positive" if positive_score > 0.6 else "Neutral" if positive_score > 0.4 else "Negative"
                main_score = positive_score
                
        # Get token-level analysis
        input_tokens = inputs['input_ids'][0]
        tokens = [model_info['tokenizer'].decode([token]) for token in input_tokens]
        
        # Remove special tokens and create text segments
        # Filter out padding, CLS, SEP, etc.
        valid_tokens = [token for token in tokens if token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']]
        
        # Get top tokens (simplified - in a real system we would use attention scores)
        top_tokens = valid_tokens[:min(10, len(valid_tokens))]
        
        breakdown = {
            "sentiment": sentiment,
            "score": main_score,
            "detailed_scores": scores,
            "model_type": model_key.capitalize() + " Transformer",
            "top_tokens": top_tokens,
            "explanation": f"Transformer models like {model_key.upper()} use self-attention to weigh the importance of different words."
        }
        
        breakdowns[model_key] = breakdown
    
    return breakdowns

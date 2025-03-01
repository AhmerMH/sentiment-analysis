from flask import Blueprint, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
import numpy as np

from app.utils import preprocess_text, validate_input
from app.models.classical_models import ClassicalModels
from app.models.deep_learning_models import DeepLearningModels
from app.models.transformer_models import TransformerModels

main = Blueprint('main', __name__)

# Initialize models
classical_models = ClassicalModels()
deep_learning_models = DeepLearningModels()
transformer_models = TransformerModels()

@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze_sentiment():
    # Get text input
    text = request.form.get('text', '')
    
    # Validate input
    valid, error_msg = validate_input(text)
    if not valid:
        return jsonify({'error': error_msg})
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Get predictions from all models
    results = {}
    
    # Classical ML models
    try:
        classical_results = classical_models.predict(processed_text)
        results.update(classical_results)
    except Exception as e:
        print(f"Error with classical models: {str(e)}")
    
    # Deep learning models
    try:
        dl_results = deep_learning_models.predict(processed_text)
        results.update(dl_results)
    except Exception as e:
        print(f"Error with deep learning models: {str(e)}")
    
    # Transformer models
    try:
        transformer_results = transformer_models.predict(text)  # Use raw text for transformers
        results.update(transformer_results)
    except Exception as e:
        print(f"Error with transformer models: {str(e)}")
    
    # Create visualization
    fig = create_sentiment_chart(results, text)
    
    # Convert plot to base64 for embedding in HTML
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    # Prepare data for rendering
    return render_template('results.html', 
                          text=text,
                          results=results,
                          plot_url=plot_url)

def create_sentiment_chart(results, text):
    """Create visualization of sentiment analysis results"""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    models = list(results.keys())
    scores = list(results.values())
    
    # Define colors based on sentiment scores
    colors = ['red' if score < 0.4 else 'yellow' if score < 0.6 else 'green' for score in scores]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(models))
    ax.barh(y_pos, scores, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Sentiment Score (0: Negative, 1: Positive)')
    ax.set_title(f'Sentiment Analysis Results for: "{text[:50]}..."')
    
    # Add value labels to bars
    for i, v in enumerate(scores):
        ax.text(v + 0.01, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    return fig

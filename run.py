
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)

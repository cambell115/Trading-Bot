import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
import numpy as np
from models import NewsArticle

try:
    from googletrans import Translator
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "googletrans==4.0.0rc1"])
    from googletrans import Translator

logger = logging.getLogger(__name__)
translation_cache = {}

class SentimentAnalyzer:
    def __init__(self):
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.translator = None

    def initialize(self) -> bool:
        try:
            logger.info("Loading FinBERT...")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
            self.translator = Translator()
            logger.info("FinBERT loaded successfully")
            return True
        except Exception as e:
            logger.error(f"FinBERT load failed: {e}")
            return False

    def analyze_sentiment_finbert(self, text: str) -> str:
        try:
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = softmax(outputs.logits, dim=-1)
            labels = ['negative', 'neutral', 'positive']
            scores = predictions[0].numpy()
            max_idx = np.argmax(scores)
            confidence = scores[max_idx]
            return labels[max_idx] if confidence > 0.6 else 'neutral'
        except Exception as e:
            logger.error(f"Sentiment error: {e}")
            return 'neutral'
    
    def analyze_article_sentiment(self, article: NewsArticle) -> str:
        """Enhanced sentiment analysis using title + description"""
        try:
            title = article.title or ''
            description = getattr(article, 'description', '') or ''
            
            # Use title + description if available
            if description and len(description.strip()) > 10:
                content = f"{title}. {description}"
            else:
                content = title
            
            return self.analyze_sentiment_finbert(content)
            
        except Exception as e:
            logger.error(f"Article sentiment analysis error: {e}")
            return 'neutral'

    async def translate_chinese_to_english(self, chinese_text: str) -> str:
        if not chinese_text or len(chinese_text.strip()) < 5:
            return chinese_text
        
        cache_key = chinese_text[:100]
        if cache_key in translation_cache:
            return translation_cache[cache_key]
        
        try:
            result = self.translator.translate(chinese_text, src='auto', dest='en')
            translated_text = result.text
            translation_cache[cache_key] = translated_text
            return translated_text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return chinese_text
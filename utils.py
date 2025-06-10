
from typing import List
from models import NewsArticle

def remove_duplicate_articles(articles: List[NewsArticle]) -> List[NewsArticle]:
    """Enhanced duplicate removal with fuzzy title matching"""
    unique_articles = []
    seen_titles = set()
    
    for article in articles:
        title_words = set(article.title.lower().split())
        is_duplicate = False
        
        for seen_title in seen_titles:
            seen_words = set(seen_title.split())
            if len(title_words) > 0 and len(seen_words) > 0:
                overlap = len(title_words.intersection(seen_words))
                similarity = overlap / min(len(title_words), len(seen_words))
                if similarity > 0.7:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_articles.append(article)
            seen_titles.add(article.title.lower())
    
    return unique_articles
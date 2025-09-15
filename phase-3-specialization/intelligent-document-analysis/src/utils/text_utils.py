"""
Text processing utility functions.
"""

import re
import string
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    try:
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        # Fallback to simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


def extract_paragraphs(text: str) -> List[str]:
    """
    Extract paragraphs from text.
    
    Args:
        text: Input text
        
    Returns:
        List of paragraphs
    """
    if not text:
        return []
    
    # Split by double newlines or multiple newlines
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def extract_words(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Extract words from text.
    
    Args:
        text: Input text
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        List of words
    """
    if not text:
        return []
    
    try:
        words = word_tokenize(text.lower())
        
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words and w.isalpha()]
        else:
            words = [w for w in words if w.isalpha()]
        
        return words
    except Exception:
        # Fallback to simple word extraction
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words]
        return words


def calculate_readability_score(text: str) -> Dict[str, float]:
    """
    Calculate readability scores for text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary containing readability metrics
    """
    if not text:
        return {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 0.0,
            'avg_sentence_length': 0.0,
            'avg_word_length': 0.0
        }
    
    try:
        # Flesch Reading Ease (0-100, higher is easier)
        flesch_ease = flesch_reading_ease(text)
        
        # Flesch-Kincaid Grade Level
        fk_grade = flesch_kincaid_grade(text)
        
        # Calculate average sentence length
        sentences = extract_sentences(text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Calculate average word length
        words = extract_words(text, remove_stopwords=False)
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        
        return {
            'flesch_reading_ease': round(flesch_ease, 2),
            'flesch_kincaid_grade': round(fk_grade, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_word_length': round(avg_word_length, 2)
        }
    except Exception:
        return {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 0.0,
            'avg_sentence_length': 0.0,
            'avg_word_length': 0.0
        }


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[Tuple[str, float]]:
    """
    Extract key phrases from text using simple frequency analysis.
    
    Args:
        text: Input text
        max_phrases: Maximum number of phrases to return
        
    Returns:
        List of tuples (phrase, score)
    """
    if not text:
        return []
    
    # Extract words
    words = extract_words(text, remove_stopwords=True)
    
    # Create bigrams and trigrams
    bigrams = []
    trigrams = []
    
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        bigrams.append(bigram)
    
    for i in range(len(words) - 2):
        trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
        trigrams.append(trigram)
    
    # Count frequencies
    phrase_counts = {}
    
    # Count bigrams
    for bigram in bigrams:
        phrase_counts[bigram] = phrase_counts.get(bigram, 0) + 1
    
    # Count trigrams
    for trigram in trigrams:
        phrase_counts[trigram] = phrase_counts.get(trigram, 0) + 1
    
    # Calculate scores (frequency * length)
    phrase_scores = []
    for phrase, count in phrase_counts.items():
        score = count * len(phrase.split())
        phrase_scores.append((phrase, score))
    
    # Sort by score and return top phrases
    phrase_scores.sort(key=lambda x: x[1], reverse=True)
    return phrase_scores[:max_phrases]


def detect_language(text: str) -> str:
    """
    Detect the language of the text (simple implementation).
    
    Args:
        text: Input text
        
    Returns:
        Detected language code
    """
    if not text:
        return "unknown"
    
    # Simple language detection based on common words
    text_lower = text.lower()
    
    # English indicators
    english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    english_count = sum(1 for word in english_words if word in text_lower)
    
    # Spanish indicators
    spanish_words = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le']
    spanish_count = sum(1 for word in spanish_words if word in text_lower)
    
    # French indicators
    french_words = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans']
    french_count = sum(1 for word in french_words if word in text_lower)
    
    # German indicators
    german_words = ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf']
    german_count = sum(1 for word in german_words if word in text_lower)
    
    # Determine language based on word counts
    counts = {
        'en': english_count,
        'es': spanish_count,
        'fr': french_count,
        'de': german_count
    }
    
    detected_lang = max(counts.items(), key=lambda x: x[1])
    
    # Return language if confidence is high enough
    if detected_lang[1] >= 3:
        return detected_lang[0]
    else:
        return "en"  # Default to English


def remove_personal_information(text: str) -> str:
    """
    Remove or mask personal information from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with personal information masked
    """
    if not text:
        return ""
    
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Phone numbers (various formats)
    text = re.sub(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})', '[PHONE]', text)
    
    # Social Security Numbers (US format)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    # Credit Card Numbers (basic pattern)
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CREDIT_CARD]', text)
    
    # IP Addresses
    text = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '[IP_ADDRESS]', text)
    
    return text


def extract_dates(text: str) -> List[str]:
    """
    Extract dates from text using regex patterns.
    
    Args:
        text: Input text
        
    Returns:
        List of found dates
    """
    if not text:
        return []
    
    # Date patterns
    patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD or YYYY-MM-DD
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',  # DD Month YYYY
    ]
    
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    
    return list(set(dates))  # Remove duplicates


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Input text
        
    Returns:
        List of found URLs
    """
    if not text:
        return []
    
    # URL pattern
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    urls = re.findall(url_pattern, text)
    return list(set(urls))  # Remove duplicates


def calculate_text_statistics(text: str) -> Dict[str, int]:
    """
    Calculate comprehensive text statistics.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary containing text statistics
    """
    if not text:
        return {
            'character_count': 0,
            'character_count_no_spaces': 0,
            'word_count': 0,
            'sentence_count': 0,
            'paragraph_count': 0,
            'line_count': 0,
            'unique_word_count': 0,
            'avg_word_length': 0.0,
            'avg_sentence_length': 0.0
        }
    
    # Basic counts
    character_count = len(text)
    character_count_no_spaces = len(text.replace(' ', ''))
    
    # Word and sentence counts
    words = extract_words(text, remove_stopwords=False)
    sentences = extract_sentences(text)
    paragraphs = extract_paragraphs(text)
    lines = text.split('\n')
    
    # Unique words
    unique_words = set(words)
    
    # Average lengths
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    
    return {
        'character_count': character_count,
        'character_count_no_spaces': character_count_no_spaces,
        'word_count': len(words),
        'sentence_count': len(sentences),
        'paragraph_count': len(paragraphs),
        'line_count': len(lines),
        'unique_word_count': len(unique_words),
        'avg_word_length': round(avg_word_length, 2),
        'avg_sentence_length': round(avg_sentence_length, 2)
    }

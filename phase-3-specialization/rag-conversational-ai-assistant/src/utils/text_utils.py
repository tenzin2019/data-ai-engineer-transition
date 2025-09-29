"""
Text Utilities Module
Provides text preprocessing, cleaning, and splitting functionality
"""

import re
import string
from typing import List, Optional
import unicodedata

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(self):
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with'
        }
    
    def process(self, text: str) -> str:
        """Main text preprocessing pipeline"""
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # Step 1: Normalize Unicode
            text = self._normalize_unicode(text)
            
            # Step 2: Clean whitespace
            text = self._clean_whitespace(text)
            
            # Step 3: Remove special characters (optional)
            text = self._clean_special_characters(text)
            
            # Step 4: Normalize case (optional)
            text = self._normalize_case(text)
            
            return text.strip()
            
        except Exception as e:
            print(f"❌ Error preprocessing text: {e}")
            return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        try:
            # Normalize to NFC form
            text = unicodedata.normalize('NFC', text)
            return text
        except Exception:
            return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace"""
        try:
            # Replace multiple whitespace with single space
            text = re.sub(r'\s+', ' ', text)
            # Remove leading/trailing whitespace
            text = text.strip()
            return text
        except Exception:
            return text
    
    def _clean_special_characters(self, text: str) -> str:
        """Remove or replace special characters"""
        try:
            # Keep letters, numbers, spaces, and common punctuation
            text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
            # Clean up any double spaces created
            text = re.sub(r'\s+', ' ', text)
            return text
        except Exception:
            return text
    
    def _normalize_case(self, text: str) -> str:
        """Normalize text case"""
        try:
            # Convert to lowercase for consistency
            return text.lower()
        except Exception:
            return text
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text"""
        try:
            # Simple keyword extraction (can be enhanced with NLP libraries)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Filter out stop words and get unique words
            keywords = [word for word in words if word not in self.stop_words]
            unique_keywords = list(set(keywords))
            
            # Return top keywords by frequency
            word_freq = {}
            for word in keywords:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_keywords[:max_keywords]]
            
        except Exception as e:
            print(f"❌ Error extracting keywords: {e}")
            return []
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags from text"""
        try:
            # Simple HTML tag removal
            text = re.sub(r'<[^>]+>', '', text)
            return text
        except Exception:
            return text
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        try:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            text = re.sub(url_pattern, '', text)
            return text
        except Exception:
            return text

class TextSplitter:
    """Text splitting utilities for chunking documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        try:
            if not text or len(text) <= self.chunk_size:
                return [text] if text else []
            
            chunks = []
            start = 0
            
            while start < len(text):
                # Calculate end position
                end = start + self.chunk_size
                
                # If this isn't the last chunk, try to break at a sentence boundary
                if end < len(text):
                    end = self._find_sentence_boundary(text, start, end)
                
                # Extract chunk
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                
                # Move start position with overlap
                start = end - self.chunk_overlap
                
                # Prevent infinite loop
                if start >= end:
                    start = end
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error splitting text: {e}")
            return [text] if text else []
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find a good sentence boundary within the chunk"""
        try:
            # Look for sentence endings within the last 200 characters
            search_start = max(start, end - 200)
            chunk_text = text[search_start:end]
            
            # Look for sentence endings
            sentence_endings = ['.', '!', '?', '\n\n']
            
            for i in range(len(chunk_text) - 1, -1, -1):
                if chunk_text[i] in sentence_endings:
                    # Check if it's followed by whitespace
                    if i + 1 < len(chunk_text) and chunk_text[i + 1].isspace():
                        return search_start + i + 1
            
            # If no sentence boundary found, return original end
            return end
            
        except Exception:
            return end
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs"""
        try:
            paragraphs = text.split('\n\n')
            chunks = []
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                if len(paragraph) <= self.chunk_size:
                    chunks.append(paragraph)
                else:
                    # Split long paragraphs using regular chunking
                    chunks.extend(self.split_text(paragraph))
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error splitting by paragraphs: {e}")
            return [text] if text else []
    
    def split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences"""
        try:
            # Simple sentence splitting (can be enhanced with NLP libraries)
            sentences = re.split(r'[.!?]+', text)
            
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # If adding this sentence would exceed chunk size, save current chunk
                if len(current_chunk) + len(sentence) + 1 > self.chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += ". " + sentence
                    else:
                        current_chunk = sentence
            
            # Add the last chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error splitting by sentences: {e}")
            return [text] if text else []

class TextMetrics:
    """Text analysis and metrics utilities"""
    
    @staticmethod
    def calculate_readability_score(text: str) -> float:
        """Calculate a simple readability score"""
        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return 0.0
            
            words = re.findall(r'\b\w+\b', text.lower())
            
            if not words:
                return 0.0
            
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Simple readability formula (can be enhanced)
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
            return max(0, min(100, score))
            
        except Exception:
            return 0.0
    
    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text"""
        try:
            words = re.findall(r'\b\w+\b', text)
            return len(words)
        except Exception:
            return 0
    
    @staticmethod
    def count_sentences(text: str) -> int:
        """Count sentences in text"""
        try:
            sentences = re.split(r'[.!?]+', text)
            return len([s for s in sentences if s.strip()])
        except Exception:
            return 0
    
    @staticmethod
    def get_text_stats(text: str) -> dict:
        """Get comprehensive text statistics"""
        try:
            return {
                'character_count': len(text),
                'word_count': TextMetrics.count_words(text),
                'sentence_count': TextMetrics.count_sentences(text),
                'readability_score': TextMetrics.calculate_readability_score(text),
                'paragraph_count': len([p for p in text.split('\n\n') if p.strip()])
            }
        except Exception:
            return {}

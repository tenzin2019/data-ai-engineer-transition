"""
Entity extraction and recognition module.
"""

import re
import logging
from typing import List, Dict, Any, Optional
import spacy
from spacy import displacy

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Handles entity extraction and recognition from text."""
    
    def __init__(self):
        """Initialize the entity extractor."""
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load the spaCy model."""
        try:
            # Try to load the English model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy English model successfully")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            # Create a basic model as fallback
            self.nlp = None
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text using spaCy.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of entity dictionaries
        """
        if not text or not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entity = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.8,  # Default confidence for spaCy
                    'description': spacy.explain(ent.label_) or ""
                }
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def extract_custom_entities(self, text: str, patterns: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Extract custom entities using regex patterns.
        
        Args:
            text: Input text
            patterns: Dictionary of entity types and regex patterns
            
        Returns:
            List of custom entities
        """
        entities = []
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = {
                    'text': match.group(),
                    'label': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.7,  # Default confidence for regex
                    'description': f"Custom {entity_type} entity"
                }
                entities.append(entity)
        
        return entities
    
    def extract_financial_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial entities from text."""
        financial_patterns = {
            'CURRENCY': r'\$[\d,]+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY)',
            'PERCENTAGE': r'\d+(?:\.\d+)?%',
            'REVENUE': r'revenue|income|sales|earnings',
            'COST': r'cost|expense|expenditure|spending',
            'PROFIT': r'profit|net income|earnings|margin',
            'ASSET': r'asset|property|investment|portfolio',
            'LIABILITY': r'liability|debt|loan|obligation'
        }
        
        return self.extract_custom_entities(text, financial_patterns)
    
    def extract_legal_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract legal entities from text."""
        legal_patterns = {
            'CONTRACT_DATE': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            'CLAUSE_NUMBER': r'clause\s+\d+|section\s+\d+|article\s+\d+',
            'LEGAL_TERM': r'\b(?:agreement|contract|license|warranty|liability|indemnity|confidentiality|termination)\b',
            'PARTY': r'\b(?:party|licensor|licensee|contractor|client|customer)\b',
            'JURISDICTION': r'\b(?:state|county|country|jurisdiction)\s+of\s+[A-Za-z\s]+\b'
        }
        
        return self.extract_custom_entities(text, legal_patterns)
    
    def extract_technical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract technical entities from text."""
        technical_patterns = {
            'API_ENDPOINT': r'https?://[^\s]+',
            'VERSION_NUMBER': r'v?\d+\.\d+(?:\.\d+)?',
            'TECHNOLOGY': r'\b(?:API|REST|GraphQL|JSON|XML|HTTP|HTTPS|OAuth|JWT)\b',
            'CODE_BLOCK': r'```[\s\S]*?```',
            'FUNCTION_NAME': r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(',
            'VARIABLE': r'\$[a-zA-Z_][a-zA-Z0-9_]*'
        }
        
        return self.extract_custom_entities(text, technical_patterns)
    
    def merge_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge overlapping or duplicate entities.
        
        Args:
            entities: List of entities to merge
            
        Returns:
            List of merged entities
        """
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda x: x['start'])
        
        merged = []
        current = entities[0]
        
        for entity in entities[1:]:
            # Check if entities overlap
            if entity['start'] <= current['end']:
                # Merge entities
                current['end'] = max(current['end'], entity['end'])
                current['text'] = current['text'] + entity['text'][current['end'] - entity['start']:]
                current['confidence'] = max(current['confidence'], entity['confidence'])
            else:
                merged.append(current)
                current = entity
        
        merged.append(current)
        return merged
    
    def get_entity_statistics(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about extracted entities.
        
        Args:
            entities: List of entities
            
        Returns:
            Dictionary containing entity statistics
        """
        if not entities:
            return {
                'total_entities': 0,
                'entity_types': {},
                'most_common_type': None,
                'average_confidence': 0.0
            }
        
        # Count entities by type
        entity_types = {}
        total_confidence = 0
        
        for entity in entities:
            entity_type = entity['label']
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            total_confidence += entity.get('confidence', 0)
        
        # Find most common type
        most_common_type = max(entity_types.items(), key=lambda x: x[1])[0] if entity_types else None
        
        return {
            'total_entities': len(entities),
            'entity_types': entity_types,
            'most_common_type': most_common_type,
            'average_confidence': total_confidence / len(entities) if entities else 0.0
        }
    
    def visualize_entities(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Create HTML visualization of entities in text.
        
        Args:
            text: Original text
            entities: List of entities
            
        Returns:
            HTML string for visualization
        """
        if not self.nlp or not entities:
            return f"<p>No entities to visualize</p>"
        
        try:
            doc = self.nlp(text)
            html = displacy.render(doc, style="ent", jupyter=False)
            return html
        except Exception as e:
            logger.error(f"Error creating entity visualization: {str(e)}")
            return f"<p>Error creating visualization: {str(e)}</p>"

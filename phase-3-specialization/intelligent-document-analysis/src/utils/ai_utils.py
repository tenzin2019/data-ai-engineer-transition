"""
AI and machine learning utility functions.
"""

import re
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings


def chunk_text(text: str, max_length: int = None, overlap: int = 200) -> List[str]:
    """
    Split text into chunks for processing by AI models.
    
    Args:
        text: Input text to chunk
        max_length: Maximum length of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    if max_length is None:
        max_length = settings.max_document_length
    
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        # If this isn't the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 200 characters
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + max_length - 200:
                end = sentence_end + 1
            else:
                # Look for word boundaries
                word_end = text.rfind(' ', start, end)
                if word_end > start + max_length - 100:
                    end = word_end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate the number of tokens in text for a given model.
    
    Args:
        text: Input text
        model: Model name for token estimation
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Rough estimation: 1 token ≈ 4 characters for English text
    # This is a simplified approach; actual tokenization varies by model
    base_ratio = 4
    
    # Adjust ratio based on model
    model_ratios = {
        "gpt-4": 4,
        "gpt-3.5-turbo": 4,
        "text-davinci-003": 4,
        "claude-3": 3.5,
        "claude-2": 3.5,
    }
    
    ratio = model_ratios.get(model, base_ratio)
    
    # Count characters and estimate tokens
    char_count = len(text)
    estimated_tokens = int(char_count / ratio)
    
    # Add some buffer for special tokens and encoding variations
    return int(estimated_tokens * 1.1)


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with provided variables.
    
    Args:
        template: Prompt template string
        **kwargs: Variables to substitute in the template
        
    Returns:
        Formatted prompt string
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required variable in prompt template: {e}")


def validate_prompt_length(prompt: str, max_tokens: int = None) -> bool:
    """
    Validate that a prompt is within token limits.
    
    Args:
        prompt: Prompt text to validate
        max_tokens: Maximum allowed tokens
        
    Returns:
        True if prompt is within limits, False otherwise
    """
    if max_tokens is None:
        max_tokens = settings.max_tokens
    
    estimated_tokens = estimate_tokens(prompt)
    return estimated_tokens <= max_tokens


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text that may contain other content.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Extracted JSON object or None
    """
    import json
    
    # Look for JSON object patterns
    json_patterns = [
        r'\{.*\}',  # Simple object
        r'\[.*\]',  # Simple array
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    return None


def clean_ai_response(response: str) -> str:
    """
    Clean AI response text by removing common artifacts.
    
    Args:
        response: Raw AI response
        
    Returns:
        Cleaned response text
    """
    if not response:
        return ""
    
    # Remove common AI response prefixes
    prefixes_to_remove = [
        "Here's the analysis:",
        "Based on the document:",
        "The analysis shows:",
        "Here is the analysis:",
        "Analysis:",
        "Summary:",
        "Here's what I found:",
    ]
    
    cleaned = response.strip()
    
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    # Remove markdown formatting if present
    cleaned = re.sub(r'```json\s*', '', cleaned)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
    
    # Remove extra whitespace
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def create_analysis_prompt(document_type: str, text_sample: str) -> str:
    """
    Create a standardized analysis prompt for different document types.
    
    Args:
        document_type: Type of document (e.g., 'legal', 'financial', 'technical')
        text_sample: Sample of the document text
        
    Returns:
        Formatted analysis prompt
    """
    
    base_prompt = """
    Please analyze the following {document_type} document and provide a comprehensive analysis.
    
    Document text:
    {text_sample}
    
    Please provide your analysis in the following JSON format:
    {{
        "summary": "A concise 2-3 sentence summary",
        "key_points": ["list", "of", "key", "points"],
        "entities": [
            {{"text": "entity", "type": "PERSON|ORGANIZATION|DATE|LOCATION", "confidence": 0.9}}
        ],
        "sentiment": {{"score": 0.5, "label": "positive|negative|neutral"}},
        "topics": ["main", "topics"],
        "recommendations": ["actionable", "recommendations"]
    }}
    """
    
    # Add document-type specific instructions
    type_instructions = {
        "legal": """
        Focus on:
        - Legal terms and clauses
        - Obligations and responsibilities
        - Compliance requirements
        - Risk factors
        - Important dates and deadlines
        """,
        "financial": """
        Focus on:
        - Financial metrics and KPIs
        - Revenue and cost information
        - Market trends and analysis
        - Investment opportunities
        - Risk assessment
        """,
        "technical": """
        Focus on:
        - Technical specifications
        - System requirements
        - Implementation details
        - Performance metrics
        - Integration points
        """,
        "medical": """
        Focus on:
        - Medical terminology
        - Diagnosis and treatment
        - Patient information
        - Medication details
        - Clinical findings
        """,
        "business": """
        Focus on:
        - Business objectives
        - Market analysis
        - Competitive landscape
        - Strategic initiatives
        - Performance indicators
        """
    }
    
    instructions = type_instructions.get(document_type.lower(), "")
    
    return base_prompt.format(
        document_type=document_type,
        text_sample=text_sample[:2000]  # Limit sample size
    ) + instructions


def calculate_confidence_score(analysis_results: Dict[str, Any]) -> float:
    """
    Calculate confidence score for analysis results.
    
    Args:
        analysis_results: Analysis results dictionary
        
    Returns:
        Confidence score between 0 and 1
    """
    confidence_factors = []
    
    # Check if summary exists and is substantial
    summary = analysis_results.get('summary', '')
    if summary and len(summary) > 50:
        confidence_factors.append(0.8)
    elif summary:
        confidence_factors.append(0.5)
    else:
        confidence_factors.append(0.2)
    
    # Check entity extraction quality
    entities = analysis_results.get('entities', [])
    if entities:
        avg_entity_confidence = sum(e.get('confidence', 0) for e in entities) / len(entities)
        confidence_factors.append(avg_entity_confidence)
    else:
        confidence_factors.append(0.3)
    
    # Check key points extraction
    key_points = analysis_results.get('key_points', [])
    if len(key_points) >= 3:
        confidence_factors.append(0.8)
    elif len(key_points) >= 1:
        confidence_factors.append(0.6)
    else:
        confidence_factors.append(0.3)
    
    # Check sentiment analysis
    sentiment = analysis_results.get('sentiment', {})
    if sentiment.get('score') is not None:
        # Higher confidence for more extreme sentiment scores
        sentiment_confidence = min(1.0, abs(sentiment.get('score', 0)) * 2 + 0.5)
        confidence_factors.append(sentiment_confidence)
    else:
        confidence_factors.append(0.4)
    
    # Calculate weighted average
    if confidence_factors:
        return sum(confidence_factors) / len(confidence_factors)
    else:
        return 0.5


def validate_analysis_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean analysis results.
    
    Args:
        results: Raw analysis results
        
    Returns:
        Validated and cleaned results
    """
    validated = {
        'summary': '',
        'key_points': [],
        'entities': [],
        'sentiment': {'score': 0.0, 'label': 'neutral'},
        'topics': [],
        'recommendations': [],
        'confidence_score': 0.0
    }
    
    # Validate summary
    summary = results.get('summary', '')
    if isinstance(summary, str) and summary.strip():
        validated['summary'] = summary.strip()
    
    # Validate key points
    key_points = results.get('key_points', [])
    if isinstance(key_points, list):
        validated['key_points'] = [str(point).strip() for point in key_points if str(point).strip()]
    
    # Validate entities
    entities = results.get('entities', [])
    if isinstance(entities, list):
        validated_entities = []
        for entity in entities:
            if isinstance(entity, dict) and 'text' in entity and 'type' in entity:
                validated_entities.append({
                    'text': str(entity['text']).strip(),
                    'type': str(entity['type']).upper(),
                    'confidence': float(entity.get('confidence', 0.5))
                })
        validated['entities'] = validated_entities
    
    # Validate sentiment
    sentiment = results.get('sentiment', {})
    if isinstance(sentiment, dict):
        score = float(sentiment.get('score', 0.0))
        label = sentiment.get('label', 'neutral')
        if label not in ['positive', 'negative', 'neutral']:
            label = 'positive' if score > 0.1 else 'negative' if score < -0.1 else 'neutral'
        validated['sentiment'] = {'score': score, 'label': label}
    
    # Validate topics
    topics = results.get('topics', [])
    if isinstance(topics, list):
        validated['topics'] = [str(topic).strip() for topic in topics if str(topic).strip()]
    
    # Validate recommendations
    recommendations = results.get('recommendations', [])
    if isinstance(recommendations, list):
        validated['recommendations'] = [str(rec).strip() for rec in recommendations if str(rec).strip()]
    
    # Calculate confidence score
    validated['confidence_score'] = calculate_confidence_score(validated)
    
    return validated


def create_comparison_prompt(doc1_text: str, doc2_text: str, comparison_type: str = "general") -> str:
    """
    Create a prompt for comparing two documents.
    
    Args:
        doc1_text: First document text
        doc2_text: Second document text
        comparison_type: Type of comparison (e.g., 'legal', 'financial')
        
    Returns:
        Formatted comparison prompt
    """
    
    prompt = f"""
    Please compare the following two {comparison_type} documents and provide a detailed comparison.
    
    Document 1:
    {doc1_text[:1500]}
    
    Document 2:
    {doc2_text[:1500]}
    
    Please provide your comparison in the following JSON format:
    {{
        "similarities": ["list", "of", "similarities"],
        "differences": ["list", "of", "differences"],
        "key_insights": ["important", "insights", "from", "comparison"],
        "recommendations": ["actionable", "recommendations"],
        "overall_assessment": "Overall assessment of the comparison"
    }}
    """
    
    return prompt

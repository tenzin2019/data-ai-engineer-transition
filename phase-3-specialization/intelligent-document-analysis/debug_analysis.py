#!/usr/bin/env python3
"""
Debug script to test the analysis functionality
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from core.ai_analyzer import AIAnalyzer

def test_fallback_analysis():
    """Test the fallback analysis with sample text."""
    
    # Sample text for testing
    sample_text = """
    This is a sample document for testing the analysis functionality.
    It contains multiple sentences to test the summary generation.
    The document includes various topics like technology, business, and innovation.
    We want to see if the fallback analysis can extract key phrases and entities.
    This should generate meaningful insights and recommendations.
    """
    
    print("🔍 Testing fallback analysis...")
    print(f"Sample text length: {len(sample_text)} characters")
    print(f"Sample text: {sample_text[:100]}...")
    print()
    
    # Initialize analyzer
    analyzer = AIAnalyzer()
    
    # Test fallback analysis
    result = analyzer._fallback_analysis(
        text=sample_text,
        document_type="general",
        include_entities=True,
        include_sentiment=True,
        include_summary=True,
        include_recommendations=True
    )
    
    print("📊 Analysis Results:")
    print(f"Summary: {result['summary']}")
    print(f"Key Phrases: {result['key_phrases']}")
    print(f"Entities: {result['entities']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Topics: {result['topics']}")
    print(f"Insights: {result['insights']}")
    print(f"Recommendations: {result['recommendations']}")
    
    # Check if results are empty
    empty_fields = []
    for key, value in result.items():
        if not value or (isinstance(value, list) and len(value) == 0):
            empty_fields.append(key)
    
    if empty_fields:
        print(f"\n⚠️ Empty fields: {empty_fields}")
    else:
        print("\n✅ All fields have content")

if __name__ == "__main__":
    test_fallback_analysis()

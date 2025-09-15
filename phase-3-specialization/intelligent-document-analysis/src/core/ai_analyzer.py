"""
AI-powered document analysis using Azure OpenAI services.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from openai import AzureOpenAI
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from ..config.settings import settings
from ..utils.text_utils import clean_text, chunk_text

logger = logging.getLogger(__name__)


class AIAnalyzer:
    """AI-powered document analysis using Azure OpenAI and Document Intelligence."""
    
    def __init__(self):
        """Initialize the AI analyzer with Azure services."""
        self.openai_client = None
        self.document_intelligence_client = None
        
        # Initialize Azure OpenAI client
        if settings.azure_openai_endpoint and settings.azure_openai_api_key:
            self.openai_client = AzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version
            )
        
        # Initialize Document Intelligence client
        if (settings.azure_document_intelligence_endpoint and 
            settings.azure_document_intelligence_api_key):
            self.document_intelligence_client = DocumentIntelligenceClient(
                endpoint=settings.azure_document_intelligence_endpoint,
                credential=AzureKeyCredential(settings.azure_document_intelligence_api_key)
            )
    
    def analyze_document(self, text: str, document_type: str = "general") -> Dict[str, Any]:
        """
        Perform comprehensive AI analysis on document text.
        
        Args:
            text: Document text content
            document_type: Type of document for context-specific analysis
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.openai_client:
            raise ValueError("Azure OpenAI client not initialized")
        
        # Clean and prepare text
        cleaned_text = clean_text(text)
        
        # Split text into chunks if too long
        chunks = chunk_text(cleaned_text, max_length=settings.max_document_length)
        
        analysis_results = {
            'summary': '',
            'key_phrases': [],
            'entities': [],
            'sentiment': {'score': 0.0, 'label': 'neutral'},
            'topics': [],
            'insights': [],
            'recommendations': [],
            'confidence_score': 0.0
        }
        
        try:
            # Process each chunk
            chunk_results = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                chunk_result = self._analyze_text_chunk(chunk, document_type)
                chunk_results.append(chunk_result)
            
            # Combine results from all chunks
            analysis_results = self._combine_chunk_results(chunk_results)
            
            # Generate final insights and recommendations
            analysis_results['insights'] = self._generate_insights(analysis_results, document_type)
            analysis_results['recommendations'] = self._generate_recommendations(analysis_results, document_type)
            
            # Calculate overall confidence score
            analysis_results['confidence_score'] = self._calculate_confidence_score(analysis_results)
            
        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}")
            raise
        
        return analysis_results
    
    def _analyze_text_chunk(self, text: str, document_type: str) -> Dict[str, Any]:
        """Analyze a single text chunk."""
        
        # Create analysis prompt based on document type
        prompt = self._create_analysis_prompt(text, document_type)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.azure_openai_deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert document analyst. Provide detailed, accurate analysis in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text chunk: {str(e)}")
            return {
                'summary': '',
                'key_phrases': [],
                'entities': [],
                'sentiment': {'score': 0.0, 'label': 'neutral'},
                'topics': [],
                'error': str(e)
            }
    
    def _create_analysis_prompt(self, text: str, document_type: str) -> str:
        """Create analysis prompt based on document type."""
        
        base_prompt = f"""
        Analyze the following {document_type} document text and provide a comprehensive analysis in JSON format.
        
        Text to analyze:
        {text[:4000]}  # Limit text length for prompt
        
        Please provide the analysis in the following JSON structure:
        {{
            "summary": "A concise 2-3 sentence summary of the document",
            "key_phrases": ["list", "of", "important", "phrases"],
            "entities": [
                {{"text": "entity name", "type": "PERSON|ORGANIZATION|DATE|LOCATION|OTHER", "confidence": 0.9}}
            ],
            "sentiment": {{"score": 0.5, "label": "positive|negative|neutral"}},
            "topics": ["main", "topics", "covered"],
            "insights": ["key", "insights", "from", "analysis"],
            "recommendations": ["actionable", "recommendations"]
        }}
        """
        
        # Add document-type specific instructions
        if document_type == "legal":
            base_prompt += """
            Focus on legal terms, clauses, obligations, and compliance requirements.
            """
        elif document_type == "financial":
            base_prompt += """
            Focus on financial data, metrics, trends, and business implications.
            """
        elif document_type == "technical":
            base_prompt += """
            Focus on technical specifications, requirements, and implementation details.
            """
        elif document_type == "medical":
            base_prompt += """
            Focus on medical terms, diagnoses, treatments, and patient information.
            """
        
        return base_prompt
    
    def _combine_chunk_results(self, chunk_results: List[Dict]) -> Dict[str, Any]:
        """Combine results from multiple text chunks."""
        
        combined = {
            'summary': '',
            'key_phrases': [],
            'entities': [],
            'sentiment': {'score': 0.0, 'label': 'neutral'},
            'topics': [],
            'insights': [],
            'recommendations': []
        }
        
        # Combine summaries
        summaries = [r.get('summary', '') for r in chunk_results if r.get('summary')]
        if summaries:
            combined['summary'] = self._summarize_summaries(summaries)
        
        # Combine key phrases (remove duplicates)
        all_phrases = []
        for result in chunk_results:
            all_phrases.extend(result.get('key_phrases', []))
        combined['key_phrases'] = list(set(all_phrases))[:20]  # Top 20 unique phrases
        
        # Combine entities (merge similar entities)
        all_entities = []
        for result in chunk_results:
            all_entities.extend(result.get('entities', []))
        combined['entities'] = self._merge_entities(all_entities)
        
        # Calculate average sentiment
        sentiments = [r.get('sentiment', {}) for r in chunk_results if r.get('sentiment')]
        if sentiments:
            avg_score = sum(s.get('score', 0) for s in sentiments) / len(sentiments)
            combined['sentiment'] = {
                'score': avg_score,
                'label': 'positive' if avg_score > 0.1 else 'negative' if avg_score < -0.1 else 'neutral'
            }
        
        # Combine topics
        all_topics = []
        for result in chunk_results:
            all_topics.extend(result.get('topics', []))
        combined['topics'] = list(set(all_topics))[:10]  # Top 10 unique topics
        
        return combined
    
    def _summarize_summaries(self, summaries: List[str]) -> str:
        """Create a final summary from multiple chunk summaries."""
        if not summaries:
            return ""
        
        if len(summaries) == 1:
            return summaries[0]
        
        # Combine summaries and create a final summary
        combined_text = " ".join(summaries)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.azure_openai_deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise summaries. Create a 2-3 sentence summary that captures the main points."},
                    {"role": "user", "content": f"Create a concise summary of the following text:\n\n{combined_text}"}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error creating final summary: {str(e)}")
            return summaries[0]  # Return first summary as fallback
    
    def _merge_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge similar entities and remove duplicates."""
        if not entities:
            return []
        
        # Group entities by text (case-insensitive)
        entity_groups = {}
        for entity in entities:
            text = entity.get('text', '').lower()
            if text not in entity_groups:
                entity_groups[text] = []
            entity_groups[text].append(entity)
        
        # Merge entities in each group
        merged_entities = []
        for text, group in entity_groups.items():
            if not group:
                continue
            
            # Use the entity with highest confidence
            best_entity = max(group, key=lambda x: x.get('confidence', 0))
            
            # Update confidence to average of all instances
            avg_confidence = sum(e.get('confidence', 0) for e in group) / len(group)
            best_entity['confidence'] = avg_confidence
            best_entity['count'] = len(group)  # Number of occurrences
            
            merged_entities.append(best_entity)
        
        # Sort by confidence and return top entities
        merged_entities.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return merged_entities[:50]  # Top 50 entities
    
    def _generate_insights(self, analysis_results: Dict, document_type: str) -> List[str]:
        """Generate insights based on analysis results."""
        insights = []
        
        # Sentiment insights
        sentiment = analysis_results.get('sentiment', {})
        if sentiment.get('score', 0) > 0.3:
            insights.append("Document shows positive sentiment overall")
        elif sentiment.get('score', 0) < -0.3:
            insights.append("Document shows negative sentiment overall")
        
        # Entity insights
        entities = analysis_results.get('entities', [])
        if entities:
            entity_types = {}
            for entity in entities:
                entity_type = entity.get('type', 'OTHER')
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            most_common_type = max(entity_types.items(), key=lambda x: x[1])
            insights.append(f"Document contains {most_common_type[1]} {most_common_type[0].lower()} entities")
        
        # Topic insights
        topics = analysis_results.get('topics', [])
        if len(topics) > 5:
            insights.append("Document covers multiple diverse topics")
        elif len(topics) <= 2:
            insights.append("Document is focused on specific topics")
        
        # Document type specific insights
        if document_type == "legal":
            insights.append("Legal document analysis completed")
        elif document_type == "financial":
            insights.append("Financial document analysis completed")
        
        return insights
    
    def _generate_recommendations(self, analysis_results: Dict, document_type: str) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Sentiment-based recommendations
        sentiment = analysis_results.get('sentiment', {})
        if sentiment.get('score', 0) < -0.3:
            recommendations.append("Consider reviewing negative aspects mentioned in the document")
        
        # Entity-based recommendations
        entities = analysis_results.get('entities', [])
        person_entities = [e for e in entities if e.get('type') == 'PERSON']
        if person_entities:
            recommendations.append(f"Follow up with {len(person_entities)} key individuals mentioned")
        
        # Topic-based recommendations
        topics = analysis_results.get('topics', [])
        if 'compliance' in [t.lower() for t in topics]:
            recommendations.append("Review compliance requirements and ensure adherence")
        
        if 'deadline' in [t.lower() for t in topics]:
            recommendations.append("Check for important deadlines and create action items")
        
        # Document type specific recommendations
        if document_type == "legal":
            recommendations.append("Consult with legal team for contract review")
        elif document_type == "financial":
            recommendations.append("Schedule financial review meeting")
        
        return recommendations
    
    def _calculate_confidence_score(self, analysis_results: Dict) -> float:
        """Calculate overall confidence score for the analysis."""
        confidence_factors = []
        
        # Entity confidence
        entities = analysis_results.get('entities', [])
        if entities:
            avg_entity_confidence = sum(e.get('confidence', 0) for e in entities) / len(entities)
            confidence_factors.append(avg_entity_confidence)
        
        # Sentiment confidence (based on score magnitude)
        sentiment = analysis_results.get('sentiment', {})
        sentiment_score = abs(sentiment.get('score', 0))
        confidence_factors.append(sentiment_score)
        
        # Content richness (based on number of key phrases and topics)
        key_phrases = len(analysis_results.get('key_phrases', []))
        topics = len(analysis_results.get('topics', []))
        content_richness = min(1.0, (key_phrases + topics) / 20)  # Normalize to 0-1
        confidence_factors.append(content_richness)
        
        # Calculate weighted average
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5  # Default confidence

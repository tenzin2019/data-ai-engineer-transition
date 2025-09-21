"""
Smart model selection utility for cost-optimized document analysis.
"""

import logging
from typing import Dict, Any, Optional
from config.settings import settings

logger = logging.getLogger(__name__)


class ModelSelector:
    """Intelligent model selection based on document characteristics and cost optimization."""
    
    def __init__(self):
        """Initialize the model selector with configuration."""
        self.primary_model = settings.primary_model
        self.secondary_model = settings.secondary_model
        self.budget_model = settings.budget_model
        self.complex_types = settings.complex_document_types
        self.budget_threshold = settings.max_tokens_budget_threshold
    
    def select_model(self, 
                    document_type: str, 
                    text_length: int, 
                    complexity_score: float = 0.5,
                    user_preference: Optional[str] = None) -> str:
        """
        Select the most appropriate model based on document characteristics.
        
        Args:
            document_type: Type of document (general, legal, financial, etc.)
            text_length: Length of document text in characters
            complexity_score: Estimated complexity (0.0 = simple, 1.0 = complex)
            user_preference: User's preferred model (overrides auto-selection)
            
        Returns:
            Selected model name
        """
        if user_preference:
            logger.info(f"Using user-preferred model: {user_preference}")
            return user_preference
        
        # Calculate estimated tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = text_length // 4
        
        # Decision logic
        if self._is_complex_document(document_type, complexity_score):
            logger.info(f"Complex document detected, using primary model: {self.primary_model}")
            return self.primary_model
        
        elif estimated_tokens <= self.budget_threshold and complexity_score < 0.3:
            logger.info(f"Simple short document, using budget model: {self.budget_model}")
            return self.budget_model
        
        else:
            logger.info(f"Standard document, using secondary model: {self.secondary_model}")
            return self.secondary_model
    
    def _is_complex_document(self, document_type: str, complexity_score: float) -> bool:
        """Determine if document requires primary model."""
        return (
            document_type.lower() in self.complex_types or 
            complexity_score > 0.7 or
            settings.use_primary_for_complex
        )
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        model_info = {
            "gpt-4o": {
                "name": "GPT-4o",
                "description": "Most capable model for complex analysis",
                "cost_per_1k_tokens": {"input": 0.005, "output": 0.015},
                "reliability": 0.95,
                "speed": "fast",
                "best_for": ["complex documents", "legal", "financial", "technical"]
            },
            "gpt-4o-mini": {
                "name": "GPT-4o Mini",
                "description": "Balanced performance and cost",
                "cost_per_1k_tokens": {"input": 0.0006, "output": 0.0024},
                "reliability": 0.90,
                "speed": "very fast",
                "best_for": ["standard documents", "general analysis", "development"]
            },
            "gpt-3.5-turbo": {
                "name": "GPT-3.5 Turbo",
                "description": "Cost-effective for simple tasks",
                "cost_per_1k_tokens": {"input": 0.0005, "output": 0.0015},
                "reliability": 0.85,
                "speed": "very fast",
                "best_for": ["simple documents", "high volume", "basic analysis"]
            }
        }
        
        return model_info.get(model_name, {
            "name": model_name,
            "description": "Unknown model",
            "cost_per_1k_tokens": {"input": 0.001, "output": 0.002},
            "reliability": 0.80,
            "speed": "unknown",
            "best_for": ["general use"]
        })
    
    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a given model and token usage."""
        model_info = self.get_model_info(model_name)
        input_cost = (input_tokens / 1000) * model_info["cost_per_1k_tokens"]["input"]
        output_cost = (output_tokens / 1000) * model_info["cost_per_1k_tokens"]["output"]
        return input_cost + output_cost
    
    def get_recommendation(self, document_type: str, text_length: int) -> Dict[str, Any]:
        """Get model recommendation with cost analysis."""
        selected_model = self.select_model(document_type, text_length)
        model_info = self.get_model_info(selected_model)
        
        # Estimate tokens and cost
        estimated_tokens = text_length // 4
        estimated_output = min(estimated_tokens // 4, 1000)  # Assume 25% output ratio, max 1000 tokens
        
        estimated_cost = self.estimate_cost(selected_model, estimated_tokens, estimated_output)
        
        return {
            "recommended_model": selected_model,
            "model_info": model_info,
            "estimated_tokens": estimated_tokens,
            "estimated_cost": estimated_cost,
            "reasoning": self._get_reasoning(document_type, text_length, selected_model)
        }
    
    def _get_reasoning(self, document_type: str, text_length: int, selected_model: str) -> str:
        """Generate human-readable reasoning for model selection."""
        if selected_model == self.primary_model:
            return f"Selected {selected_model} for complex {document_type} document with {text_length} characters"
        elif selected_model == self.budget_model:
            return f"Selected {selected_model} for simple document with {text_length} characters (cost-optimized)"
        else:
            return f"Selected {selected_model} for standard document analysis"


# Global model selector instance
model_selector = ModelSelector()

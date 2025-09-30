"""
LLM Orchestrator Module
Manages multiple LLM providers and handles model selection, load balancing, and fallback
"""

import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

# LLM providers
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

# LangChain for orchestration
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

from ..utils.text_utils import TextPreprocessor

class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, name: str, model: str, api_key: str):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.is_available = True
        self.last_used = None
        self.error_count = 0
        self.max_errors = 3
    
    async def generate_response(self, prompt: str, max_tokens: int = 500, 
                              temperature: float = 0.7, **kwargs) -> str:
        """Generate response - to be implemented by subclasses"""
        raise NotImplementedError
    
    def mark_error(self):
        """Mark an error occurred"""
        self.error_count += 1
        if self.error_count >= self.max_errors:
            self.is_available = False
            print(f"Warning: Provider {self.name} marked as unavailable due to errors")
    
    def mark_success(self):
        """Mark successful usage"""
        self.error_count = 0
        self.is_available = True
        self.last_used = datetime.now()

class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__("openai", model, api_key)
        self.client = AsyncOpenAI(api_key=api_key)
        self.langchain_llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.7
        )
    
    async def generate_response(self, prompt: str, max_tokens: int = 500, 
                              temperature: float = 0.7, **kwargs) -> str:
        """Generate response using OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            self.mark_success()
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI error: {e}")
            self.mark_error()
            raise

class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation"""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        super().__init__("anthropic", model, api_key)
        self.client = AsyncAnthropic(api_key=api_key)
        self.langchain_llm = ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=0.7
        )
    
    async def generate_response(self, prompt: str, max_tokens: int = 500, 
                              temperature: float = 0.7, **kwargs) -> str:
        """Generate response using Anthropic"""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            
            self.mark_success()
            return response.content[0].text
            
        except Exception as e:
            print(f"Anthropic error: {e}")
            self.mark_error()
            raise

class LLMOrchestrator:
    """Main LLM orchestrator that manages multiple providers"""
    
    def __init__(self):
        self.providers: List[LLMProvider] = []
        self.text_preprocessor = TextPreprocessor()
        self.conversation_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize providers
        self._initialize_providers()
        
        print("LLM Orchestrator initialized")
    
    def _initialize_providers(self):
        """Initialize available LLM providers"""
        try:
            # OpenAI provider
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                openai_provider = OpenAIProvider(openai_key)
                self.providers.append(openai_provider)
                print(f"OpenAI provider initialized: {openai_provider.model}")
            
            # Anthropic provider
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key:
                anthropic_provider = AnthropicProvider(anthropic_key)
                self.providers.append(anthropic_provider)
                print(f"Anthropic provider initialized: {anthropic_provider.model}")
            
            if not self.providers:
                print("Warning: No LLM providers configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables.")
                
        except Exception as e:
            print(f"Error initializing providers: {e}")
    
    async def generate_response(self, prompt: str, max_tokens: int = 500, 
                              temperature: float = 0.7, conversation_id: Optional[str] = None,
                              preferred_provider: Optional[str] = None, **kwargs) -> str:
        """Generate response using available providers"""
        try:
            if not self.providers:
                raise ValueError("No LLM providers available")
            
            # Select provider
            provider = self._select_provider(preferred_provider)
            
            if not provider:
                raise ValueError("No available providers")
            
            # Add conversation context if available
            enhanced_prompt = await self._enhance_prompt_with_context(prompt, conversation_id)
            
            # Generate response
            start_time = time.time()
            response = await provider.generate_response(
                prompt=enhanced_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Cache conversation
            if conversation_id:
                await self._cache_conversation(conversation_id, prompt, response)
            
            # Log performance
            response_time = time.time() - start_time
            print(f"Generated response using {provider.name} in {response_time:.2f}s")
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            # Try fallback providers
            return await self._try_fallback_generation(prompt, max_tokens, temperature, **kwargs)
    
    def _select_provider(self, preferred_provider: Optional[str] = None) -> Optional[LLMProvider]:
        """Select the best available provider"""
        try:
            # Filter available providers
            available_providers = [p for p in self.providers if p.is_available]
            
            if not available_providers:
                return None
            
            # If preferred provider is specified and available
            if preferred_provider:
                for provider in available_providers:
                    if provider.name == preferred_provider:
                        return provider
            
            # Simple load balancing - select provider with least recent usage
            # or least error count
            best_provider = min(available_providers, key=lambda p: (
                p.error_count,
                p.last_used.timestamp() if p.last_used else 0
            ))
            
            return best_provider
            
        except Exception as e:
            print(f"Error selecting provider: {e}")
            return available_providers[0] if available_providers else None
    
    async def _try_fallback_generation(self, prompt: str, max_tokens: int, 
                                     temperature: float, **kwargs) -> str:
        """Try generating response with fallback providers"""
        try:
            # Get all available providers except the one that failed
            available_providers = [p for p in self.providers if p.is_available]
            
            if not available_providers:
                return "I apologize, but I'm currently unable to generate a response. Please try again later."
            
            # Try each available provider
            for provider in available_providers:
                try:
                    response = await provider.generate_response(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs
                    )
                    
                    provider.mark_success()
                    print(f"Fallback successful using {provider.name}")
                    return response
                    
                except Exception as e:
                    print(f"Fallback provider {provider.name} failed: {e}")
                    provider.mark_error()
                    continue
            
            # All providers failed
            return "I apologize, but I'm currently experiencing technical difficulties. Please try again later."
            
        except Exception as e:
            print(f"All fallback attempts failed: {e}")
            return "I apologize, but I'm currently unable to process your request. Please try again later."
    
    async def _enhance_prompt_with_context(self, prompt: str, conversation_id: Optional[str] = None) -> str:
        """Enhance prompt with conversation context"""
        try:
            if not conversation_id or conversation_id not in self.conversation_cache:
                return prompt
            
            # Get recent conversation history
            history = self.conversation_cache[conversation_id][-3:]  # Last 3 interactions
            
            if not history:
                return prompt
            
            # Build context
            context_parts = []
            for interaction in history:
                context_parts.append(f"Previous: {interaction['user']}")
                context_parts.append(f"Response: {interaction['assistant']}")
            
            context = "\n".join(context_parts)
            enhanced_prompt = f"Context from previous conversation:\n{context}\n\nCurrent question: {prompt}"
            
            return enhanced_prompt
            
        except Exception as e:
            print(f"Error enhancing prompt: {e}")
            return prompt
    
    async def _cache_conversation(self, conversation_id: str, user_input: str, assistant_response: str):
        """Cache conversation for context"""
        try:
            if conversation_id not in self.conversation_cache:
                self.conversation_cache[conversation_id] = []
            
            interaction = {
                'user': user_input,
                'assistant': assistant_response,
                'timestamp': datetime.now().isoformat()
            }
            
            self.conversation_cache[conversation_id].append(interaction)
            
            # Keep only last 10 interactions
            if len(self.conversation_cache[conversation_id]) > 10:
                self.conversation_cache[conversation_id] = self.conversation_cache[conversation_id][-10:]
                
        except Exception as e:
            print(f"Error caching conversation: {e}")
    
    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        try:
            status = {
                'total_providers': len(self.providers),
                'available_providers': len([p for p in self.providers if p.is_available]),
                'providers': []
            }
            
            for provider in self.providers:
                provider_status = {
                    'name': provider.name,
                    'model': provider.model,
                    'is_available': provider.is_available,
                    'error_count': provider.error_count,
                    'last_used': provider.last_used.isoformat() if provider.last_used else None
                }
                status['providers'].append(provider_status)
            
            return status
            
        except Exception as e:
            print(f"Error getting provider status: {e}")
            return {'error': str(e)}
    
    async def reset_provider_errors(self, provider_name: Optional[str] = None):
        """Reset error counts for providers"""
        try:
            for provider in self.providers:
                if not provider_name or provider.name == provider_name:
                    provider.error_count = 0
                    provider.is_available = True
                    print(f"Reset errors for provider: {provider.name}")
                    
        except Exception as e:
            print(f"Error resetting provider errors: {e}")
    
    async def clear_conversation_cache(self, conversation_id: Optional[str] = None):
        """Clear conversation cache"""
        try:
            if conversation_id:
                if conversation_id in self.conversation_cache:
                    del self.conversation_cache[conversation_id]
                    print(f"Cleared cache for conversation: {conversation_id}")
            else:
                self.conversation_cache.clear()
                print("Cleared all conversation caches")
                
        except Exception as e:
            print(f"Error clearing conversation cache: {e}")

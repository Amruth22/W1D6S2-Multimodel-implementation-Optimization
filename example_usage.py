#!/usr/bin/env python3
"""
Example usage of the Multimodal Engine library with prompt engineering
"""

import os
from multimodal_engine import MultimodalEngine, MultimodalContent, ModalityType
from prompt_engineering import PromptStyle, PromptContext, PromptTemplate

def example_text_generation():
    """Example of text generation with prompt engineering"""
    print("=== Text Generation with Prompt Engineering ===")
    
    try:
        engine = MultimodalEngine()
        
        # Simple text generation
        result = engine.generate_from_text("Write a short poem about artificial intelligence")
        print(f"Simple generation: {result.text[:100]}...")
        print(f"Response time: {result.response_time:.2f}s")
        print()
        
        # Text generation with style enhancement
        result = engine.generate_from_text(
            "Explain quantum computing", 
            style=PromptStyle.TECHNICAL,
            target_audience="university students",
            max_length=150
        )
        print(f"Technical style: {result.text[:100]}...")
        print(f"Response time: {result.response_time:.2f}s")
        print()
        
        # Creative style
        result = engine.generate_from_text(
            "Tell me a story about a robot", 
            style=PromptStyle.CREATIVE
        )
        print(f"Creative style: {result.text[:100]}...")
        print(f"Response time: {result.response_time:.2f}s")
        print()
        
    except Exception as e:
        print(f"Error in text generation: {e}")

def example_image_analysis():
    """Example of image analysis"""
    print("=== Image Analysis Example ===")
    
    try:
        engine = MultimodalEngine()
        
        # For demonstration, we'll show the API usage
        print("Note: This example shows the API usage. For actual image analysis, provide real image files.")
        
        # Example with prompt engineering
        result = engine.generate_from_image(
            "dummy_image_data",  # Placeholder
            prompt="What's in this image?",
            style=PromptStyle.DETAILED,
            target_audience="art historians"
        )
        print("Image analysis with prompt engineering ready.")
        print()
        
    except Exception as e:
        print(f"Error in image analysis: {e}")

def example_audio_analysis():
    """Example of audio analysis"""
    print("=== Audio Analysis Example ===")
    
    try:
        engine = MultimodalEngine()
        
        # For demonstration, we'll show the API usage
        print("Note: This example shows the API usage. For actual audio analysis, provide real audio files.")
        
        # Example with prompt engineering
        result = engine.generate_from_audio(
            "dummy_audio_data",  # Placeholder
            prompt="Summarize this audio",
            style=PromptStyle.CONCISE,
            target_audience="business professionals"
        )
        print("Audio analysis with prompt engineering ready.")
        print()
        
    except Exception as e:
        print(f"Error in audio analysis: {e}")

def example_prompt_templates():
    """Example of using prompt templates"""
    print("=== Prompt Templates Example ===")
    
    try:
        engine = MultimodalEngine()
        
        # Show how templates work (conceptually)
        print("Prompt templates available:")
        print("- IMAGE_DESCRIPTION: Detailed image analysis")
        print("- AUDIO_TRANSCRIPTION: Audio to text conversion")
        print("- COMPARISON: Compare multiple inputs")
        print("- SENTIMENT: Analyze emotional tone")
        print()
        
        print("To use templates, integrate them with the prompt_engineer in the MultimodalEngine.")
        print()
        
    except Exception as e:
        print(f"Error in prompt templates: {e}")

def example_multimodal_analysis():
    """Example of multimodal analysis"""
    print("=== Multimodal Analysis Example ===")
    
    try:
        engine = MultimodalEngine()
        
        # Create multimodal content
        contents = [
            MultimodalContent(
                modality=ModalityType.TEXT,
                data="Machine learning is a subset of artificial intelligence that focuses on algorithms that improve automatically through experience."
            )
        ]
        
        result = engine.generate_from_multimodal(contents, "Analyze these inputs and explain how they relate to each other")
        print(f"Multimodal analysis: {result.text[:100]}...")
        print(f"Response time: {result.response_time:.2f}s")
        print(f"Cached: {result.cached}")
        print()
        
    except Exception as e:
        print(f"Error in multimodal analysis: {e}")

def example_optimization_features():
    """Example of optimization features"""
    print("=== Optimization Features Example ===")
    
    try:
        engine = MultimodalEngine()
        
        # Enable caching
        engine.enable_caching = True
        print(f"Caching enabled: {engine.enable_caching}")
        
        # Enable compression
        engine.enable_compression = True
        print(f"Compression enabled: {engine.enable_compression}")
        
        # Generate content (first time, not cached)
        result1 = engine.generate_from_text("What is the capital of France?")
        print(f"First request - Response time: {result1.response_time:.2f}s, Cached: {result1.cached}")
        
        # Generate the same content (should be cached)
        result2 = engine.generate_from_text("What is the capital of France?")
        print(f"Second request - Response time: {result2.response_time:.2f}s, Cached: {result2.cached}")
        
        # Check cache statistics
        stats = engine.get_cache_stats()
        print(f"Cache statistics: {stats}")
        
        # Clear cache
        engine.clear_cache()
        print("Cache cleared")
        
        stats = engine.get_cache_stats()
        print(f"Cache statistics after clearing: {stats}")
        print()
        
    except Exception as e:
        print(f"Error in optimization features: {e}")

if __name__ == "__main__":
    # Run examples
    example_text_generation()
    example_image_analysis()
    example_audio_analysis()
    example_prompt_templates()
    example_multimodal_analysis()
    example_optimization_features()
    
    print("All examples completed!")
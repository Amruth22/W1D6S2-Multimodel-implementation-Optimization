#!/usr/bin/env python3
"""
Prompt Engineering Module for Multimodal Engine

This module provides utilities for optimizing prompts for different modalities
and implementing advanced prompt engineering techniques.
"""

from typing import List, Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
import re

class PromptStyle(Enum):
    """Different styles of prompts"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    CONCISE = "concise"
    DETAILED = "detailed"

class PromptTemplate(Enum):
    """Predefined prompt templates"""
    IMAGE_DESCRIPTION = "Describe this image in detail, including colors, objects, and context."
    IMAGE_OCR = "Extract all text visible in this image."
    AUDIO_TRANSCRIPTION = "Transcribe the spoken content in this audio clip."
    AUDIO_SUMMARY = "Summarize the key points from this audio clip."
    COMPARISON = "Compare and contrast the provided inputs, highlighting similarities and differences."
    SENTIMENT = "Analyze the sentiment expressed in this content."

@dataclass
class PromptContext:
    """Context for prompt engineering"""
    style: PromptStyle
    target_audience: str = "general"
    max_length: Optional[int] = None
    keywords: Optional[List[str]] = None
    language: str = "en"

class PromptEngineer:
    """Advanced prompt engineering utilities"""
    
    def __init__(self):
        """Initialize the prompt engineer"""
        self.templates = {
            PromptTemplate.IMAGE_DESCRIPTION: "Describe this image in detail, including colors, objects, and context.",
            PromptTemplate.IMAGE_OCR: "Extract all text visible in this image.",
            PromptTemplate.AUDIO_TRANSCRIPTION: "Transcribe the spoken content in this audio clip.",
            PromptTemplate.AUDIO_SUMMARY: "Summarize the key points from this audio clip.",
            PromptTemplate.COMPARISON: "Compare and contrast the provided inputs, highlighting similarities and differences.",
            PromptTemplate.SENTIMENT: "Analyze the sentiment expressed in this content."
        }
    
    def enhance_prompt(self, base_prompt: str, context: PromptContext) -> str:
        """
        Enhance a prompt based on the provided context
        
        Args:
            base_prompt: The original prompt
            context: PromptContext with enhancement parameters
            
        Returns:
            Enhanced prompt string
        """
        enhanced = base_prompt
        
        # Apply style modifications
        if context.style == PromptStyle.ANALYTICAL:
            enhanced = f"Analyze the following content analytically: {base_prompt}"
        elif context.style == PromptStyle.CREATIVE:
            enhanced = f"Respond creatively to: {base_prompt}"
        elif context.style == PromptStyle.TECHNICAL:
            enhanced = f"Provide a technical explanation of: {base_prompt}"
        elif context.style == PromptStyle.CONCISE:
            enhanced = f"Give a concise response to: {base_prompt}"
        elif context.style == PromptStyle.DETAILED:
            enhanced = f"Provide a detailed response to: {base_prompt}"
        
        # Add audience context
        if context.target_audience != "general":
            enhanced = f"For a {context.target_audience} audience: {enhanced}"
        
        # Add keywords if provided
        if context.keywords:
            keywords_str = ", ".join(context.keywords)
            enhanced = f"{enhanced} Focus on these keywords: {keywords_str}"
        
        # Apply length constraints
        if context.max_length:
            enhanced = f"{enhanced} Keep your response under {context.max_length} words."
        
        return enhanced
    
    def apply_template(self, template: PromptTemplate, customizations: Optional[Dict[str, Any]] = None) -> str:
        """
        Apply a predefined template with optional customizations
        
        Args:
            template: The template to apply
            customizations: Optional dictionary of customizations
            
        Returns:
            Template-based prompt string
        """
        if template not in self.templates:
            raise ValueError(f"Unknown template: {template}")
        
        prompt = self.templates[template]
        
        # Apply customizations if provided
        if customizations:
            for key, value in customizations.items():
                placeholder = f"{{{key}}}"
                if placeholder in prompt:
                    prompt = prompt.replace(placeholder, str(value))
        
        return prompt
    
    def create_multimodal_prompt(self, modalities: List[str], task: str, context: Optional[PromptContext] = None) -> str:
        """
        Create a prompt for multimodal analysis
        
        Args:
            modalities: List of modality types (text, image, audio)
            task: The task to perform
            context: Optional context for prompt engineering
            
        Returns:
            Multimodal prompt string
        """
        modality_str = ", ".join(modalities)
        prompt = f"Analyze the following {modality_str} inputs and {task}"
        
        if context:
            prompt = self.enhance_prompt(prompt, context)
            
        return prompt
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text for prompt enhancement
        
        Args:
            text: Input text to analyze
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction (in a real implementation, you might use NLP libraries)
        # Remove common stop words and punctuation
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words and get unique words
        keywords = list(set(word for word in words if word not in stop_words))
        
        # Return top keywords (limited by max_keywords)
        return keywords[:max_keywords]
    
    def optimize_for_modality(self, prompt: str, modality: str) -> str:
        """
        Optimize a prompt for a specific modality
        
        Args:
            prompt: The base prompt
            modality: The modality type (text, image, audio)
            
        Returns:
            Modality-optimized prompt
        """
        optimizations = {
            "image": f"Visually analyze: {prompt}",
            "audio": f"Aurally analyze: {prompt}",
            "text": f"Textually analyze: {prompt}"
        }
        
        return optimizations.get(modality, prompt)

# Predefined prompt engineers for common use cases
class ImagePromptEngineer(PromptEngineer):
    """Specialized prompt engineer for image analysis"""
    
    def describe_image(self, focus_areas: Optional[List[str]] = None) -> str:
        """Create a detailed image description prompt"""
        base_prompt = "Describe this image in detail"
        
        if focus_areas:
            areas = ", ".join(focus_areas)
            base_prompt += f", focusing on {areas}"
        
        return base_prompt + ". Include objects, colors, composition, and context."
    
    def extract_text(self) -> str:
        """Create an OCR prompt"""
        return "Extract all readable text from this image. Include text in any languages present."
    
    def analyze_sentiment(self) -> str:
        """Create an image sentiment analysis prompt"""
        return "Analyze the emotional tone and sentiment conveyed by this image. Consider colors, expressions, and composition."

class AudioPromptEngineer(PromptEngineer):
    """Specialized prompt engineer for audio analysis"""
    
    def transcribe_audio(self) -> str:
        """Create an audio transcription prompt"""
        return "Provide a complete transcription of all spoken content in this audio clip. Include speaker identification if possible."
    
    def summarize_audio(self) -> str:
        """Create an audio summary prompt"""
        return "Summarize the key points and main ideas from this audio clip. Highlight important details and conclusions."
    
    def analyze_tone(self) -> str:
        """Create an audio tone analysis prompt"""
        return "Analyze the tone, emotion, and delivery style of the speakers in this audio clip. Include observations about pace, volume, and inflection."

# Example usage
def example_usage():
    """Example usage of the prompt engineering module"""
    print("=== Prompt Engineering Examples ===")
    
    # Basic prompt engineering
    engineer = PromptEngineer()
    
    # Enhance a prompt with context
    context = PromptContext(
        style=PromptStyle.DETAILED,
        target_audience="students",
        max_length=100,
        keywords=["machine learning", "neural networks"]
    )
    
    base_prompt = "Explain artificial intelligence"
    enhanced = engineer.enhance_prompt(base_prompt, context)
    print(f"Enhanced prompt: {enhanced}")
    
    # Use templates
    image_desc_prompt = engineer.apply_template(PromptTemplate.IMAGE_DESCRIPTION)
    print(f"Image description template: {image_desc_prompt}")
    
    # Extract keywords
    sample_text = "Machine learning and artificial intelligence are transforming the technology landscape with neural networks and deep learning algorithms."
    keywords = engineer.extract_keywords(sample_text)
    print(f"Extracted keywords: {keywords}")
    
    # Specialized engineers
    image_engineer = ImagePromptEngineer()
    image_prompt = image_engineer.describe_image(["people", "architecture"])
    print(f"Specialized image prompt: {image_prompt}")
    
    audio_engineer = AudioPromptEngineer()
    audio_prompt = audio_engineer.transcribe_audio()
    print(f"Specialized audio prompt: {audio_prompt}")

if __name__ == "__main__":
    example_usage()
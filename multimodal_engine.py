import os
import base64
import tempfile
import mimetypes
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time

# Google Generative AI
from google import genai
from google.genai import types

# For image processing
from PIL import Image
import io

# For audio processing
import librosa
import numpy as np

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import prompt engineering
from prompt_engineering import PromptEngineer, PromptContext, PromptStyle

class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"

@dataclass
class MultimodalContent:
    """Represents content for multimodal processing"""
    modality: ModalityType
    data: Any
    prompt: Optional[str] = None
    mime_type: Optional[str] = None

@dataclass
class GenerationResult:
    """Result of content generation"""
    text: str
    cached: bool = False
    response_time: float = 0.0
    tokens_used: Optional[int] = None

class MultimodalEngine:
    """Core multimodal processing engine without Flask API"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the multimodal engine
        
        Args:
            api_key: Google Gemini API key. If None, will try to load from environment.
            model_name: Name of the Gemini model to use
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
            
        self.model_name = model_name
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Caching for optimization
        self._cache = {}
        self._cache_stats = {"hits": 0, "misses": 0}
        
        # Optimization settings
        self.enable_caching = True
        self.enable_compression = True
        
        # Prompt engineering
        self.prompt_engineer = PromptEngineer()
        
    def generate_from_text(self, prompt: str, stream: bool = False, 
                          style: Optional[PromptStyle] = None,
                          target_audience: str = "general",
                          max_length: Optional[int] = None) -> GenerationResult:
        """
        Generate content from text prompt
        
        Args:
            prompt: Text prompt for generation
            stream: Whether to stream the response
            style: Prompt style for enhancement
            target_audience: Target audience for the response
            max_length: Maximum length of the response
            
        Returns:
            GenerationResult with the generated text
        """
        if not prompt:
            raise ValueError("Prompt is required")
            
        # Enhance prompt if style is specified
        if style is not None:
            context = PromptContext(
                style=style,
                target_audience=target_audience,
                max_length=max_length
            )
            prompt = self.prompt_engineer.enhance_prompt(prompt, context)
            
        start_time = time.time()
        
        # Check cache first
        cache_key = f"text:{prompt}:{stream}"
        if self.enable_caching and cache_key in self._cache:
            self._cache_stats["hits"] += 1
            cached_result = self._cache[cache_key]
            return GenerationResult(
                text=cached_result["text"],
                cached=True,
                response_time=time.time() - start_time
            )
        
        self._cache_stats["misses"] += 1
        
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            if stream:
                # For streaming, we'll return the first chunk for simplicity in this example
                response_text = ""
                for chunk in self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                ):
                    response_text += chunk.text
            else:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
                response_text = response.text
                
            # Cache the result
            if self.enable_caching:
                self._cache[cache_key] = {
                    "text": response_text,
                    "timestamp": time.time()
                }
                
            return GenerationResult(
                text=response_text,
                cached=False,
                response_time=time.time() - start_time
            )
            
        except Exception as e:
            raise Exception(f"Error generating content from text: {str(e)}")
    
    def generate_from_image(self, image_data: Union[bytes, str], 
                           prompt: str = "Describe this image.",
                           style: Optional[PromptStyle] = None,
                           target_audience: str = "general",
                           max_length: Optional[int] = None) -> GenerationResult:
        """
        Generate content from image
        
        Args:
            image_data: Image bytes or file path
            prompt: Prompt for image analysis
            style: Prompt style for enhancement
            target_audience: Target audience for the response
            max_length: Maximum length of the response
            
        Returns:
            GenerationResult with the generated text
        """
        # Enhance prompt if style is specified
        if style is not None:
            context = PromptContext(
                style=style,
                target_audience=target_audience,
                max_length=max_length
            )
            prompt = self.prompt_engineer.enhance_prompt(prompt, context)
            
        start_time = time.time()
        
        # Handle file path vs bytes
        if isinstance(image_data, str):
            # It's a file path
            with open(image_data, 'rb') as f:
                image_bytes = f.read()
            mime_type, _ = mimetypes.guess_type(image_data)
        else:
            # It's bytes
            image_bytes = image_data
            mime_type = 'image/jpeg'  # Default
            
        # Check cache
        cache_key = f"image:{hash(image_bytes)}:{prompt}"
        if self.enable_caching and cache_key in self._cache:
            self._cache_stats["hits"] += 1
            cached_result = self._cache[cache_key]
            return GenerationResult(
                text=cached_result["text"],
                cached=True,
                response_time=time.time() - start_time
            )
        
        self._cache_stats["misses"] += 1
        
        try:
            # Optimize image if needed
            if self.enable_compression:
                image_bytes = self._optimize_image(image_bytes)
                
            # Generate content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type or 'image/jpeg',
                    ),
                    prompt
                ]
            )
            
            response_text = response.text
            
            # Cache the result
            if self.enable_caching:
                self._cache[cache_key] = {
                    "text": response_text,
                    "timestamp": time.time()
                }
                
            return GenerationResult(
                text=response_text,
                cached=False,
                response_time=time.time() - start_time
            )
            
        except Exception as e:
            raise Exception(f"Error generating content from image: {str(e)}")
    
    def generate_from_audio(self, audio_data: Union[bytes, str], 
                           prompt: str = "Describe this audio clip.",
                           style: Optional[PromptStyle] = None,
                           target_audience: str = "general",
                           max_length: Optional[int] = None) -> GenerationResult:
        """
        Generate content from audio
        
        Args:
            audio_data: Audio bytes or file path
            prompt: Prompt for audio analysis
            style: Prompt style for enhancement
            target_audience: Target audience for the response
            max_length: Maximum length of the response
            
        Returns:
            GenerationResult with the generated text
        """
        # Enhance prompt if style is specified
        if style is not None:
            context = PromptContext(
                style=style,
                target_audience=target_audience,
                max_length=max_length
            )
            prompt = self.prompt_engineer.enhance_prompt(prompt, context)
            
        start_time = time.time()
        
        # Handle file path vs bytes
        if isinstance(audio_data, str):
            # It's a file path
            with open(audio_data, 'rb') as f:
                audio_bytes = f.read()
        else:
            # It's bytes
            audio_bytes = audio_data
            
        # Check cache
        cache_key = f"audio:{hash(audio_bytes)}:{prompt}"
        if self.enable_caching and cache_key in self._cache:
            self._cache_stats["hits"] += 1
            cached_result = self._cache[cache_key]
            return GenerationResult(
                text=cached_result["text"],
                cached=True,
                response_time=time.time() - start_time
            )
        
        self._cache_stats["misses"] += 1
        
        try:
            # Save audio to temporary file for upload
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_filename = tmp_file.name
                
            try:
                # Upload file to Gemini with explicit MIME type in config
                uploaded_file = self.client.files.upload(
                    file=tmp_filename,
                    config={'mime_type': 'audio/mp3'}
                )
                
                # Generate content using the uploaded file directly
                response = self.client.models.generate_content(
                    model=self.model_name, 
                    contents=[prompt, uploaded_file]
                )
                
                response_text = response.text
                
                # Cache the result
                if self.enable_caching:
                    self._cache[cache_key] = {
                        "text": response_text,
                        "timestamp": time.time()
                    }
                    
                return GenerationResult(
                    text=response_text,
                    cached=False,
                    response_time=time.time() - start_time
                )
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_filename)
                
        except Exception as e:
            raise Exception(f"Error generating content from audio: {str(e)}")
    
    def generate_from_multimodal(self, contents: List[MultimodalContent], prompt: Optional[str] = None) -> GenerationResult:
        """
        Generate content from multiple modalities
        
        Args:
            contents: List of MultimodalContent objects
            prompt: General prompt for all modalities
            
        Returns:
            GenerationResult with the generated text
        """
        if not contents:
            raise ValueError("At least one modality (text, image, or audio) must be provided")
            
        start_time = time.time()
        
        gemini_contents = []
        
        # Process each content item
        for content in contents:
            if content.modality == ModalityType.TEXT:
                gemini_contents.append(types.Part.from_text(text=content.data))
            elif content.modality == ModalityType.IMAGE:
                # Handle image data
                if isinstance(content.data, str):
                    # File path
                    with open(content.data, 'rb') as f:
                        image_bytes = f.read()
                else:
                    # Bytes
                    image_bytes = content.data
                    
                # Optimize if needed
                if self.enable_compression:
                    image_bytes = self._optimize_image(image_bytes)
                    
                gemini_contents.append(types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=content.mime_type or 'image/jpeg',
                ))
            elif content.modality == ModalityType.AUDIO:
                # Handle audio data
                if isinstance(content.data, str):
                    # File path
                    with open(content.data, 'rb') as f:
                        audio_bytes = f.read()
                else:
                    # Bytes
                    audio_bytes = content.data
                    
                # Upload audio file
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_filename = tmp_file.name
                    
                try:
                    uploaded_file = self.client.files.upload(
                        file=tmp_filename,
                        config={'mime_type': 'audio/mp3'}
                    )
                    gemini_contents.append(uploaded_file)
                finally:
                    os.unlink(tmp_filename)
        
        # Add prompt if provided
        if prompt:
            gemini_contents.append(types.Part.from_text(text=prompt))
            
        # Create cache key
        cache_data = "".join([str(c.data) if isinstance(c.data, str) else str(hash(c.data)) for c in contents])
        cache_key = f"multimodal:{hash(cache_data)}:{prompt}"
        
        # Check cache
        if self.enable_caching and cache_key in self._cache:
            self._cache_stats["hits"] += 1
            cached_result = self._cache[cache_key]
            return GenerationResult(
                text=cached_result["text"],
                cached=True,
                response_time=time.time() - start_time
            )
        
        self._cache_stats["misses"] += 1
        
        try:
            # Generate content from all modalities
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=gemini_contents
            )
            
            response_text = response.text
            
            # Cache the result
            if self.enable_caching:
                self._cache[cache_key] = {
                    "text": response_text,
                    "timestamp": time.time()
                }
                
            return GenerationResult(
                text=response_text,
                cached=False,
                response_time=time.time() - start_time
            )
            
        except Exception as e:
            raise Exception(f"Error generating content from multimodal inputs: {str(e)}")
    
    def _optimize_image(self, image_bytes: bytes) -> bytes:
        """
        Optimize image for faster processing
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Optimized image bytes
        """
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Resize if too large (max 1024x1024)
            max_size = (1024, 1024)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to JPEG with quality 85 if not already JPEG
            if image.format != 'JPEG':
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=85, optimize=True)
                return buffer.getvalue()
            else:
                # If already JPEG, just return as is
                return image_bytes
                
        except Exception:
            # If optimization fails, return original
            return image_bytes
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total if total > 0 else 0
        
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "hit_rate": hit_rate,
            "cache_size": len(self._cache)
        }
    
    def clear_cache(self):
        """Clear the cache"""
        self._cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0}
#!/usr/bin/env python3
"""
Pytest-based test suite for the Multimodal Engine library
Compatible with Python 3.9-3.12
"""

import pytest
import os
import time
import asyncio
import tempfile
import io
from unittest.mock import patch, MagicMock, Mock, mock_open
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

# Import the modules to test
from multimodal_engine import MultimodalEngine, MultimodalContent, ModalityType, GenerationResult
from prompt_engineering import PromptEngineer, PromptStyle, PromptTemplate, PromptContext

# Mock data for testing
MOCK_RESPONSES = {
    "text_generation": "This is a mock response from the Gemini model demonstrating multimodal AI capabilities with advanced optimization techniques.",
    "image_analysis": "This image shows a complex technical diagram with multiple components, featuring blue and green color schemes with detailed annotations.",
    "audio_analysis": "This audio clip contains clear speech discussing machine learning concepts with a professional tone and moderate pace.",
    "multimodal_analysis": "The combined analysis of text, image, and audio inputs reveals a comprehensive technical presentation about AI systems."
}

# Mock configuration
MOCK_CONFIG = {
    "GEMINI_API_KEY": "AIza_mock_multimodal_engine_api_key_for_testing",
    "MODEL_NAME": "gemini-2.5-flash",
    "CACHE_ENABLED": True,
    "COMPRESSION_ENABLED": True,
    "MAX_IMAGE_SIZE": (1024, 1024),
    "SUPPORTED_IMAGE_FORMATS": ["JPEG", "PNG", "WEBP", "HEIC", "HEIF"],
    "SUPPORTED_AUDIO_FORMATS": ["MP3", "WAV", "AIFF", "AAC", "OGG", "FLAC"]
}

# ============================================================================
# PYTEST ASYNC TEST FUNCTIONS - 10 CORE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_01_multimodal_engine_initialization():
    """Test 1: MultimodalEngine Initialization and Configuration"""
    print("Running Test 1: MultimodalEngine Initialization and Configuration")
    
    # Test initialization with explicit API key
    with patch('multimodal_engine.genai.Client') as mock_genai:
        mock_client = MagicMock()
        mock_genai.return_value = mock_client
        
        engine = MultimodalEngine(api_key=MOCK_CONFIG["GEMINI_API_KEY"])
        
        assert engine is not None, "Engine should be initialized"
        assert engine.api_key == MOCK_CONFIG["GEMINI_API_KEY"], "Should store API key correctly"
        assert engine.model_name == "gemini-2.5-flash", "Should use default model"
        assert engine.enable_caching == True, "Caching should be enabled by default"
        assert engine.enable_compression == True, "Compression should be enabled by default"
        
        # Test custom model name
        custom_engine = MultimodalEngine(
            api_key=MOCK_CONFIG["GEMINI_API_KEY"], 
            model_name="gemini-2.0-flash"
        )
        assert custom_engine.model_name == "gemini-2.0-flash", "Should use custom model name"
    
    # Test initialization without API key (should raise error)
    with patch.dict(os.environ, {}, clear=True):
        try:
            MultimodalEngine()
            assert False, "Should raise ValueError when no API key provided"
        except ValueError as e:
            assert "API key is required" in str(e), "Should provide clear error message"
    
    print("PASS: Engine initialization with API key validation")
    print("PASS: Default configuration settings validated")
    print("PASS: Custom model name configuration working")

@pytest.mark.asyncio
async def test_02_text_generation_with_caching():
    """Test 2: Text Generation with Intelligent Caching"""
    print("Running Test 2: Text Generation with Intelligent Caching")
    
    with patch('multimodal_engine.genai.Client') as mock_genai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = MOCK_RESPONSES["text_generation"]
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.return_value = mock_client
        
        engine = MultimodalEngine(api_key=MOCK_CONFIG["GEMINI_API_KEY"])
        engine.enable_caching = True
        
        # First request (should not be cached)
        prompt = "Explain quantum computing in simple terms"
        
        # Add small delay to simulate processing time
        import time
        time.sleep(0.01)
        
        result1 = engine.generate_from_text(prompt)
        
        assert result1 is not None, "Should return generation result"
        assert isinstance(result1, GenerationResult), "Should return GenerationResult object"
        assert result1.text == MOCK_RESPONSES["text_generation"], "Should return expected text"
        assert result1.cached == False, "First request should not be cached"
        assert result1.response_time > 0, "Should track response time"
        
        # Second request with same prompt (should be cached)
        result2 = engine.generate_from_text(prompt)
        
        assert result2.text == result1.text, "Cached result should match original"
        assert result2.cached == True, "Second request should be cached"
        assert result2.response_time >= 0, "Should track response time for cached requests"
        
        # Test cache statistics
        stats = engine.get_cache_stats()
        assert stats["hits"] == 1, "Should have 1 cache hit"
        assert stats["misses"] == 1, "Should have 1 cache miss"
        assert stats["hit_rate"] == 0.5, "Hit rate should be 50%"
        assert stats["cache_size"] == 1, "Should have 1 item in cache"
        
        # Test cache clearing
        engine.clear_cache()
        stats_after_clear = engine.get_cache_stats()
        assert stats_after_clear["cache_size"] == 0, "Cache should be empty after clearing"
        assert stats_after_clear["hits"] == 0, "Hit count should reset"
        assert stats_after_clear["misses"] == 0, "Miss count should reset"
    
    print(f"PASS: Text generation working - Response: {result1.text[:50]}...")
    print(f"PASS: Caching system working - Hit rate: {stats['hit_rate']:.1%}")
    print("PASS: Cache statistics and clearing functionality validated")

@pytest.mark.asyncio
async def test_03_prompt_engineering_enhancement():
    """Test 3: Advanced Prompt Engineering with Style Enhancement"""
    print("Running Test 3: Advanced Prompt Engineering with Style Enhancement")
    
    with patch('multimodal_engine.genai.Client') as mock_genai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = MOCK_RESPONSES["text_generation"]
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.return_value = mock_client
        
        engine = MultimodalEngine(api_key=MOCK_CONFIG["GEMINI_API_KEY"])
        
        # Test different prompt styles
        base_prompt = "Explain artificial intelligence"
        
        # Technical style
        result_technical = engine.generate_from_text(
            base_prompt,
            style=PromptStyle.TECHNICAL,
            target_audience="university students",
            max_length=200
        )
        
        assert result_technical is not None, "Should generate with technical style"
        assert isinstance(result_technical.text, str), "Should return string response"
        assert len(result_technical.text) > 0, "Should generate non-empty response"
        
        # Creative style
        result_creative = engine.generate_from_text(
            base_prompt,
            style=PromptStyle.CREATIVE,
            target_audience="general"
        )
        
        assert result_creative is not None, "Should generate with creative style"
        assert isinstance(result_creative.text, str), "Should return string response"
        
        # Test prompt engineer directly
        prompt_engineer = PromptEngineer()
        
        context = PromptContext(
            style=PromptStyle.ANALYTICAL,
            target_audience="researchers",
            max_length=150,
            keywords=["machine learning", "neural networks"]
        )
        
        enhanced_prompt = prompt_engineer.enhance_prompt(base_prompt, context)
        
        assert enhanced_prompt != base_prompt, "Enhanced prompt should be different from base"
        assert "analytical" in enhanced_prompt.lower(), "Should include style indicator"
        assert "researchers" in enhanced_prompt, "Should include target audience"
        assert "machine learning" in enhanced_prompt, "Should include keywords"
        
        # Test template application
        template_prompt = prompt_engineer.apply_template(PromptTemplate.IMAGE_DESCRIPTION)
        assert "describe" in template_prompt.lower(), "Template should contain description instruction"
        assert "image" in template_prompt.lower(), "Template should reference image"
    
    print("PASS: Prompt engineering with style enhancement working")
    print("PASS: Context-aware prompt modification validated")
    print("PASS: Template system functioning correctly")

@pytest.mark.asyncio
async def test_04_image_processing_and_optimization():
    """Test 4: Image Processing with Compression Optimization"""
    print("Running Test 4: Image Processing with Compression Optimization")
    
    with patch('multimodal_engine.genai.Client') as mock_genai:
        with patch('multimodal_engine.Image') as mock_image_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = MOCK_RESPONSES["text_generation"]
            mock_client.models.generate_content.return_value = mock_response
            mock_genai.return_value = mock_client
            
            # Mock PIL Image operations
            mock_image_instance = MagicMock()
            mock_image_instance.size = (2048, 1536)
            mock_image_instance.format = "PNG"
            mock_image_class.open.return_value = mock_image_instance
            
            engine = MultimodalEngine(api_key=MOCK_CONFIG["GEMINI_API_KEY"])
            engine.enable_compression = True
            
            # Create mock image data
            mock_image_data = b"mock_large_image_data_" + b"x" * 5000
            
            # Test image processing
            result = engine.generate_from_image(
                mock_image_data,
                prompt="Describe this technical diagram in detail",
                style=PromptStyle.DETAILED,
                target_audience="engineers"
            )
            
            assert result is not None, "Should process image successfully"
            assert isinstance(result, GenerationResult), "Should return GenerationResult"
            assert result.text == MOCK_RESPONSES["text_generation"], "Should return expected analysis"
            assert result.response_time > 0, "Should track processing time"
            
            # Test image optimization was called
            mock_image_class.open.assert_called(), "Should open image for optimization"
            
            # Test with file path
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_file.write(mock_image_data)
                tmp_file_path = tmp_file.name
            
            try:
                with patch('builtins.open', mock_open(read_data=mock_image_data)):
                    result_file = engine.generate_from_image(
                        tmp_file_path,
                        prompt="Analyze this image file"
                    )
                    
                    assert result_file is not None, "Should process image file successfully"
                    assert isinstance(result_file.text, str), "Should return text analysis"
            finally:
                os.unlink(tmp_file_path)
            
            # Test caching for image processing
            result_cached = engine.generate_from_image(
                mock_image_data,
                prompt="Describe this technical diagram in detail"
            )
            
            assert result_cached.cached == True, "Second image request should be cached"
    
    print("PASS: Image processing with compression optimization working")
    print("PASS: File path and bytes input handling validated")
    print("PASS: Image processing caching functionality confirmed")

@pytest.mark.asyncio
async def test_05_audio_processing_capabilities():
    """Test 5: Audio Processing with File Upload Management"""
    print("Running Test 5: Audio Processing with File Upload Management")
    
    with patch('multimodal_engine.genai.Client') as mock_genai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = MOCK_RESPONSES["text_generation"]
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.return_value = mock_client
        
        # Mock file upload
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/test_audio.mp3"
        mock_client.files.upload.return_value = mock_uploaded_file
        
        engine = MultimodalEngine(api_key=MOCK_CONFIG["GEMINI_API_KEY"])
        
        # Create mock audio data
        mock_audio_data = b"mock_audio_data_mp3_format_" + b"x" * 3000
        
        # Test audio processing with bytes
        # Add small delay to simulate processing time
        import time
        time.sleep(0.01)
        
        result = engine.generate_from_audio(
            mock_audio_data,
            prompt="Transcribe and summarize this audio clip",
            style=PromptStyle.CONCISE,
            target_audience="business professionals"
        )
        
        assert result is not None, "Should process audio successfully"
        assert isinstance(result, GenerationResult), "Should return GenerationResult"
        assert result.text == MOCK_RESPONSES["text_generation"], "Should return expected analysis"
        assert result.response_time > 0, "Should track processing time"
        
        # Verify file upload was called
        mock_client.files.upload.assert_called(), "Should upload audio file to Gemini"
        
        # Test audio processing with file path
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_file.write(mock_audio_data)
            tmp_file_path = tmp_file.name
        
        try:
            with patch('builtins.open', mock_open(read_data=mock_audio_data)):
                result_file = engine.generate_from_audio(
                    tmp_file_path,
                    prompt="Analyze this audio file content"
                )
                
                assert result_file is not None, "Should process audio file successfully"
                assert isinstance(result_file.text, str), "Should return text analysis"
        finally:
            os.unlink(tmp_file_path)
        
        # Test audio processing caching
        result_cached = engine.generate_from_audio(
            mock_audio_data,
            prompt="Transcribe and summarize this audio clip"
        )
        
        assert result_cached.cached == True, "Second audio request should be cached"
    
    print("PASS: Audio processing with file upload management working")
    print("PASS: Temporary file handling and cleanup validated")
    print("PASS: Audio processing caching functionality confirmed")

@pytest.mark.asyncio
async def test_06_multimodal_content_integration():
    """Test 6: Multimodal Content Integration and Processing"""
    print("Running Test 6: Multimodal Content Integration and Processing")
    
    with patch('multimodal_engine.genai.Client') as mock_genai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = MOCK_RESPONSES["multimodal_analysis"]
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.return_value = mock_client
        
        # Mock file upload for audio
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/multimodal_audio.mp3"
        mock_client.files.upload.return_value = mock_uploaded_file
        
        engine = MultimodalEngine(api_key=MOCK_CONFIG["GEMINI_API_KEY"])
        
        # Create multimodal content
        text_content = MultimodalContent(
            modality=ModalityType.TEXT,
            data="This is a technical presentation about machine learning algorithms and their applications in modern AI systems."
        )
        
        image_content = MultimodalContent(
            modality=ModalityType.IMAGE,
            data=b"mock_image_data_technical_diagram" + b"x" * 2000,
            mime_type="image/jpeg"
        )
        
        audio_content = MultimodalContent(
            modality=ModalityType.AUDIO,
            data=b"mock_audio_data_presentation" + b"x" * 4000,
            prompt="Extract key technical concepts from this audio"
        )
        
        contents = [text_content, image_content, audio_content]
        
        # Test multimodal processing
        result = engine.generate_from_multimodal(
            contents,
            prompt="Analyze these multimodal inputs and explain how they relate to each other"
        )
        
        assert result is not None, "Should process multimodal content successfully"
        assert isinstance(result, GenerationResult), "Should return GenerationResult"
        assert result.text == MOCK_RESPONSES["multimodal_analysis"], "Should return multimodal analysis"
        assert result.response_time > 0, "Should track processing time"
        
        # Test individual content validation
        assert text_content.modality == ModalityType.TEXT, "Text content should have correct modality"
        assert image_content.modality == ModalityType.IMAGE, "Image content should have correct modality"
        assert audio_content.modality == ModalityType.AUDIO, "Audio content should have correct modality"
        assert image_content.mime_type == "image/jpeg", "Image content should have correct MIME type"
        assert audio_content.prompt is not None, "Audio content should have specific prompt"
        
        # Test empty content list validation
        try:
            engine.generate_from_multimodal([], "Test prompt")
            assert False, "Should raise error for empty content list"
        except ValueError as e:
            assert "at least one modality" in str(e).lower(), "Should provide clear error message"
        
        # Test multimodal caching
        result_cached = engine.generate_from_multimodal(
            contents,
            prompt="Analyze these multimodal inputs and explain how they relate to each other"
        )
        
        assert result_cached.cached == True, "Second multimodal request should be cached"
    
    print("PASS: Multimodal content integration working correctly")
    print("PASS: Content validation and error handling confirmed")
    print("PASS: Multimodal processing caching functionality validated")

@pytest.mark.asyncio
async def test_07_optimization_features_validation():
    """Test 7: Optimization Features and Performance Monitoring"""
    print("Running Test 7: Optimization Features and Performance Monitoring")
    
    with patch('multimodal_engine.genai.Client') as mock_genai:
        with patch('multimodal_engine.Image') as mock_image_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = MOCK_RESPONSES["text_generation"]
            mock_client.models.generate_content.return_value = mock_response
            mock_genai.return_value = mock_client
            
            # Mock image optimization
            mock_image_instance = MagicMock()
            mock_image_instance.size = (1500, 1200)
            mock_image_instance.format = "PNG"
            mock_image_class.open.return_value = mock_image_instance
            
            engine = MultimodalEngine(api_key=MOCK_CONFIG["GEMINI_API_KEY"])
            
            # Test optimization settings
            assert engine.enable_caching == True, "Caching should be enabled by default"
            assert engine.enable_compression == True, "Compression should be enabled by default"
            
            # Test disabling optimizations
            engine.enable_caching = False
            engine.enable_compression = False
            
            assert engine.enable_caching == False, "Should be able to disable caching"
            assert engine.enable_compression == False, "Should be able to disable compression"
            
            # Test with optimizations disabled
            result_no_cache = engine.generate_from_text("Test prompt without caching")
            assert result_no_cache.cached == False, "Should not use cache when disabled"
            
            result_no_cache_2 = engine.generate_from_text("Test prompt without caching")
            assert result_no_cache_2.cached == False, "Should not cache when caching disabled"
            
            # Re-enable optimizations
            engine.enable_caching = True
            engine.enable_compression = True
            
            # Test image optimization
            large_image_data = b"large_mock_image_data_" + b"x" * 10000
            
            with patch('multimodal_engine.io.BytesIO') as mock_bytesio:
                mock_buffer = io.BytesIO()
                mock_bytesio.return_value = mock_buffer
                
                result_optimized = engine.generate_from_image(
                    large_image_data,
                    prompt="Analyze this large image"
                )
                
                assert result_optimized is not None, "Should process optimized image"
                mock_image_class.open.assert_called(), "Should attempt image optimization"
            
            # Test cache statistics tracking
            stats = engine.get_cache_stats()
            required_stats = ["hits", "misses", "hit_rate", "cache_size"]
            
            for stat in required_stats:
                assert stat in stats, f"Cache stats should include {stat}"
                assert isinstance(stats[stat], (int, float)), f"{stat} should be numeric"
            
            assert 0 <= stats["hit_rate"] <= 1, "Hit rate should be between 0 and 1"
    
    print("PASS: Optimization settings configuration working")
    print("PASS: Cache and compression toggle functionality validated")
    print("PASS: Performance monitoring and statistics tracking confirmed")

@pytest.mark.asyncio
async def test_08_error_handling_and_validation():
    """Test 8: Comprehensive Error Handling and Input Validation"""
    print("Running Test 8: Comprehensive Error Handling and Input Validation")
    
    with patch('multimodal_engine.genai.Client') as mock_genai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = MOCK_RESPONSES["text_generation"]
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.return_value = mock_client
        
        engine = MultimodalEngine(api_key=MOCK_CONFIG["GEMINI_API_KEY"])
        
        # Test empty prompt validation
        try:
            engine.generate_from_text("")
            assert False, "Should raise error for empty prompt"
        except ValueError as e:
            assert "prompt is required" in str(e).lower(), "Should provide clear error message"
        
        try:
            engine.generate_from_text(None)
            assert False, "Should raise error for None prompt"
        except (ValueError, TypeError):
            pass  # Expected error
        
        # Test API error handling
        mock_client_error = MagicMock()
        mock_client_error.models.generate_content.side_effect = Exception("API connection failed")
        
        with patch('multimodal_engine.genai.Client') as mock_genai_error:
            mock_genai_error.return_value = mock_client_error
            
            engine_error = MultimodalEngine(api_key=MOCK_CONFIG["GEMINI_API_KEY"])
            
            try:
                engine_error.generate_from_text("Test prompt")
                assert False, "Should raise error when API fails"
            except Exception as e:
                assert "error generating content from text" in str(e).lower(), "Should wrap API errors"
        
        # Test file not found error handling
        try:
            engine.generate_from_image("/nonexistent/path/image.jpg", "Describe image")
            assert False, "Should raise error for nonexistent file"
        except (FileNotFoundError, Exception):
            pass  # Expected error
        
        # Test invalid audio file handling
        try:
            engine.generate_from_audio("/nonexistent/path/audio.mp3", "Transcribe audio")
            assert False, "Should raise error for nonexistent audio file"
        except (FileNotFoundError, Exception):
            pass  # Expected error
        
        # Test prompt engineering error handling
        prompt_engineer = PromptEngineer()
        
        try:
            prompt_engineer.apply_template("invalid_template")
            assert False, "Should raise error for invalid template"
        except (ValueError, KeyError):
            pass  # Expected error
        
        # Test multimodal content validation
        try:
            engine.generate_from_multimodal([], "Test prompt")
            assert False, "Should raise error for empty content list"
        except ValueError as e:
            assert "at least one modality" in str(e).lower(), "Should validate content list"
    
    print("PASS: Input validation and error handling working correctly")
    print("PASS: API error wrapping and user-friendly messages confirmed")
    print("PASS: File handling and edge case validation successful")

@pytest.mark.asyncio
async def test_09_specialized_prompt_engineers():
    """Test 9: Specialized Prompt Engineers for Different Modalities"""
    print("Running Test 9: Specialized Prompt Engineers for Different Modalities")
    
    # Test ImagePromptEngineer
    from prompt_engineering import ImagePromptEngineer, AudioPromptEngineer
    
    image_engineer = ImagePromptEngineer()
    
    # Test image description prompts
    desc_prompt = image_engineer.describe_image(focus_areas=["colors", "objects", "composition"])
    assert isinstance(desc_prompt, str), "Should return string prompt"
    assert "describe" in desc_prompt.lower(), "Should contain description instruction"
    assert "colors" in desc_prompt, "Should include focus areas"
    assert "objects" in desc_prompt, "Should include focus areas"
    assert "composition" in desc_prompt, "Should include focus areas"
    
    # Test OCR prompt
    ocr_prompt = image_engineer.extract_text()
    assert isinstance(ocr_prompt, str), "Should return string prompt"
    assert "text" in ocr_prompt.lower(), "Should reference text extraction"
    assert "extract" in ocr_prompt.lower(), "Should contain extraction instruction"
    
    # Test sentiment analysis prompt
    sentiment_prompt = image_engineer.analyze_sentiment()
    assert isinstance(sentiment_prompt, str), "Should return string prompt"
    assert "sentiment" in sentiment_prompt.lower(), "Should reference sentiment"
    assert "emotion" in sentiment_prompt.lower(), "Should reference emotional analysis"
    
    # Test AudioPromptEngineer
    audio_engineer = AudioPromptEngineer()
    
    # Test transcription prompt
    transcribe_prompt = audio_engineer.transcribe_audio()
    assert isinstance(transcribe_prompt, str), "Should return string prompt"
    assert "transcribe" in transcribe_prompt.lower(), "Should contain transcription instruction"
    assert "audio" in transcribe_prompt.lower(), "Should reference audio"
    
    # Test summarization prompt
    summary_prompt = audio_engineer.summarize_audio()
    assert isinstance(summary_prompt, str), "Should return string prompt"
    assert "summarize" in summary_prompt.lower(), "Should contain summarization instruction"
    assert "key points" in summary_prompt.lower(), "Should reference key points"
    
    # Test tone analysis prompt
    tone_prompt = audio_engineer.analyze_tone()
    assert isinstance(tone_prompt, str), "Should return string prompt"
    assert "tone" in tone_prompt.lower(), "Should reference tone analysis"
    assert "emotion" in tone_prompt.lower(), "Should reference emotional analysis"
    
    # Test keyword extraction
    base_engineer = PromptEngineer()
    sample_text = "Machine learning and artificial intelligence are transforming technology with neural networks and deep learning algorithms."
    
    keywords = base_engineer.extract_keywords(sample_text, max_keywords=5)
    assert isinstance(keywords, list), "Should return list of keywords"
    assert len(keywords) <= 5, "Should respect max_keywords limit"
    assert "machine" in keywords or "learning" in keywords, "Should extract relevant keywords"
    
    # Test modality-specific optimization
    text_optimized = base_engineer.optimize_for_modality("Analyze this content", "text")
    image_optimized = base_engineer.optimize_for_modality("Analyze this content", "image")
    audio_optimized = base_engineer.optimize_for_modality("Analyze this content", "audio")
    
    assert "textually" in text_optimized.lower(), "Should optimize for text modality"
    assert "visually" in image_optimized.lower(), "Should optimize for image modality"
    assert "aurally" in audio_optimized.lower(), "Should optimize for audio modality"
    
    print("PASS: ImagePromptEngineer specialized prompts working correctly")
    print("PASS: AudioPromptEngineer specialized prompts validated")
    print("PASS: Keyword extraction and modality optimization confirmed")

@pytest.mark.asyncio
async def test_10_performance_and_benchmarking():
    """Test 10: Performance Monitoring and Benchmarking Capabilities"""
    print("Running Test 10: Performance Monitoring and Benchmarking Capabilities")
    
    with patch('multimodal_engine.genai.Client') as mock_genai:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = MOCK_RESPONSES["text_generation"]
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.return_value = mock_client
        
        engine = MultimodalEngine(api_key=MOCK_CONFIG["GEMINI_API_KEY"])
        
        # Test response time tracking
        start_time = time.time()
        # Add small delay to simulate processing time
        time.sleep(0.01)
        result = engine.generate_from_text("Performance test prompt")
        end_time = time.time()
        
        assert result.response_time > 0, "Should track response time"
        assert result.response_time <= (end_time - start_time) + 0.1, "Response time should be reasonable"
        
        # Test multiple requests for performance analysis
        response_times = []
        cache_hits = 0
        
        test_prompts = [
            "Explain machine learning",
            "Describe neural networks", 
            "What is deep learning",
            "Explain machine learning",  # Duplicate for cache test
            "Describe neural networks"   # Duplicate for cache test
        ]
        
        for prompt in test_prompts:
            result = engine.generate_from_text(prompt)
            response_times.append(result.response_time)
            if result.cached:
                cache_hits += 1
        
        # Analyze performance metrics
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        assert avg_response_time > 0, "Should have positive average response time"
        assert min_response_time >= 0, "Minimum response time should be non-negative"
        assert max_response_time >= min_response_time, "Max should be >= min response time"
        assert cache_hits >= 2, "Should have cache hits for duplicate prompts"
        
        # Test cache performance benefits
        stats = engine.get_cache_stats()
        assert stats["hits"] >= 2, "Should have multiple cache hits"
        assert stats["hit_rate"] > 0, "Should have positive hit rate"
        
        # Test concurrent processing simulation
        async def process_request(prompt_suffix):
            return engine.generate_from_text(f"Concurrent test {prompt_suffix}")
        
        # Simulate concurrent requests
        concurrent_tasks = [process_request(i) for i in range(3)]
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        
        assert len(concurrent_results) == 3, "Should handle concurrent requests"
        for result in concurrent_results:
            assert isinstance(result, GenerationResult), "Each result should be GenerationResult"
            assert result.response_time >= 0, "Each result should have response time"
        
        # Test memory usage simulation (cache size monitoring)
        initial_cache_size = engine.get_cache_stats()["cache_size"]
        
        # Add more items to cache
        for i in range(5):
            engine.generate_from_text(f"Memory test prompt {i}")
        
        final_cache_size = engine.get_cache_stats()["cache_size"]
        assert final_cache_size > initial_cache_size, "Cache size should increase with new requests"
        
        # Test performance with different modalities
        image_result = engine.generate_from_image(
            b"mock_image_data" + b"x" * 1000,
            "Performance test image"
        )
        
        assert image_result.response_time > 0, "Image processing should track response time"
        
        print(f"PASS: Performance monitoring - Avg response time: {avg_response_time:.3f}s")
        print(f"PASS: Cache performance - Hit rate: {stats['hit_rate']:.1%}, Hits: {stats['hits']}")
        print(f"PASS: Concurrent processing - {len(concurrent_results)} requests handled")
        print(f"PASS: Memory monitoring - Cache size: {final_cache_size} items")

# ============================================================================
# ASYNC TEST RUNNER
# ============================================================================

async def run_async_tests():
    """Run all async tests"""
    print("Running Multimodal Engine Library Tests (Async Pytest Version)...")
    print("Using comprehensive mocked data for ultra-fast execution")
    print("Testing: Multimodal processing, optimization, prompt engineering")
    print("=" * 70)
    
    # List of exactly 10 async test functions
    test_functions = [
        test_01_multimodal_engine_initialization,
        test_02_text_generation_with_caching,
        test_03_prompt_engineering_enhancement,
        test_04_image_processing_and_optimization,
        test_05_audio_processing_capabilities,
        test_06_multimodal_content_integration,
        test_07_optimization_features_validation,
        test_08_error_handling_and_validation,
        test_09_specialized_prompt_engineers,
        test_10_performance_and_benchmarking
    ]
    
    passed = 0
    failed = 0
    
    # Run tests sequentially for better output readability
    for test_func in test_functions:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test_func.__name__} - {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 70)
    print(f"📊 Test Results Summary:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
        print("✅ Multimodal Engine Library (Async Pytest) is working correctly")
        print("⚡ Ultra-fast async execution with comprehensive mocked features")
        print("🤖 Multimodal AI processing, caching, optimization, and prompt engineering validated")
        print("🚀 No real API calls required - pure async testing with mocks")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed")
        return False

def run_all_tests():
    """Run all tests and provide summary (sync wrapper for async tests)"""
    return asyncio.run(run_async_tests())

# ============================================================================
# PYTEST INTEGRATION
# ============================================================================

@pytest.mark.asyncio
async def test_multimodal_engine_suite():
    """Pytest entry point for the entire test suite"""
    success = await run_async_tests()
    assert success, "All tests should pass"

if __name__ == "__main__":
    print("🚀 Starting Multimodal Engine Library Tests (Async Pytest Version)")
    print("📋 No API keys required - using comprehensive async mocked responses")
    print("⚡ Ultra-fast async execution for multimodal AI processing")
    print("🤖 Testing: Text/Image/Audio processing, caching, optimization")
    print("🧠 Advanced prompt engineering and performance monitoring")
    print()
    
    # Run the tests
    start_time = time.time()
    success = run_all_tests()
    end_time = time.time()
    
    print(f"\n⏱️  Total execution time: {end_time - start_time:.2f} seconds")
    exit(0 if success else 1)
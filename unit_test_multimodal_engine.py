#!/usr/bin/env python3
"""
Test suite for the Multimodal Engine library
"""

import unittest
import os
import sys
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Add the current directory to the path so we can import the module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from multimodal_engine import MultimodalEngine, MultimodalContent, ModalityType, GenerationResult
from prompt_engineering import PromptEngineer, PromptStyle, PromptTemplate

class TestMultimodalEngineStructure(unittest.TestCase):
    """Test the basic structure and imports of the library"""
    
    def test_imports(self):
        """Test that all modules can be imported"""
        try:
            from multimodal_engine import MultimodalEngine, ModalityType, GenerationResult
            from prompt_engineering import PromptEngineer, PromptStyle, PromptTemplate
        except Exception as e:
            self.fail(f"Import error: {e}")
    
    def test_data_classes(self):
        """Test that data classes work correctly"""
        # Test MultimodalContent
        content = MultimodalContent(
            modality=ModalityType.TEXT,
            data="Test data"
        )
        self.assertEqual(content.modality, ModalityType.TEXT)
        self.assertEqual(content.data, "Test data")
        
        # Test GenerationResult
        result = GenerationResult(
            text="Test response",
            cached=False,
            response_time=0.1
        )
        self.assertEqual(result.text, "Test response")
        self.assertEqual(result.cached, False)
        self.assertEqual(result.response_time, 0.1)

class TestMultimodalEngineUnit(unittest.TestCase):
    """Unit tests for the MultimodalEngine using mocks"""
    
    def test_init_with_api_key_from_env(self):
        """Test initialization with API key from environment"""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'}):
            engine = MultimodalEngine()
            self.assertIsNotNone(engine)
            self.assertEqual(engine.api_key, 'test_api_key')
    
    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises ValueError"""
        # Remove GEMINI_API_KEY from environment if it exists
        env_backup = os.environ.get('GEMINI_API_KEY')
        if 'GEMINI_API_KEY' in os.environ:
            del os.environ['GEMINI_API_KEY']
            
        try:
            with self.assertRaises(ValueError) as context:
                MultimodalEngine()
            self.assertIn("API key is required", str(context.exception))
        finally:
            # Restore environment
            if env_backup:
                os.environ['GEMINI_API_KEY'] = env_backup
    
    def test_init_with_explicit_api_key(self):
        """Test initialization with explicit API key"""
        engine = MultimodalEngine(api_key="explicit_test_key")
        self.assertEqual(engine.api_key, "explicit_test_key")
    
    @patch('multimodal_engine.genai.Client')
    def test_generate_from_text_success(self, mock_client_class):
        """Test successful text generation"""
        # Mock the Gemini client
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "This is a test response from the AI model."
        mock_client_instance.models.generate_content.return_value = mock_response
        
        engine = MultimodalEngine(api_key="test_key")
        result = engine.generate_from_text("Test prompt")
        
        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.text, "This is a test response from the AI model.")
        self.assertFalse(result.cached)
        self.assertGreater(result.response_time, 0)
    
    @patch('multimodal_engine.genai.Client')
    def test_generate_from_text_with_caching(self, mock_client_class):
        """Test text generation with caching"""
        # Mock the Gemini client
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "This is a test response from the AI model."
        mock_client_instance.models.generate_content.return_value = mock_response
        
        engine = MultimodalEngine(api_key="test_key")
        engine.enable_caching = True
        
        # First call
        result1 = engine.generate_from_text("Test prompt")
        self.assertFalse(result1.cached)
        
        # Second call (should be cached)
        result2 = engine.generate_from_text("Test prompt")
        self.assertTrue(result2.cached)
        self.assertEqual(result1.text, result2.text)
    
    def test_generate_from_text_empty_prompt(self):
        """Test text generation with empty prompt raises ValueError"""
        engine = MultimodalEngine(api_key="test_key")
        
        with self.assertRaises(ValueError) as context:
            engine.generate_from_text("")
        self.assertEqual(str(context.exception), "Prompt is required")
    
    @patch('multimodal_engine.genai.Client')
    def test_generate_from_text_exception_handling(self, mock_client_class):
        """Test text generation exception handling"""
        # Mock the Gemini client to raise an exception
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.models.generate_content.side_effect = Exception("API Error")
        
        engine = MultimodalEngine(api_key="test_key")
        
        with self.assertRaises(Exception) as context:
            engine.generate_from_text("Test prompt")
        self.assertIn("Error generating content from text", str(context.exception))
    
    def test_multimodal_content_creation(self):
        """Test creation of MultimodalContent objects"""
        # Test text content
        text_content = MultimodalContent(
            modality=ModalityType.TEXT,
            data="Test text content"
        )
        self.assertEqual(text_content.modality, ModalityType.TEXT)
        self.assertEqual(text_content.data, "Test text content")
        
        # Test image content
        image_content = MultimodalContent(
            modality=ModalityType.IMAGE,
            data=b"fake_image_data",
            mime_type="image/jpeg"
        )
        self.assertEqual(image_content.modality, ModalityType.IMAGE)
        self.assertEqual(image_content.data, b"fake_image_data")
        self.assertEqual(image_content.mime_type, "image/jpeg")
        
        # Test audio content
        audio_content = MultimodalContent(
            modality=ModalityType.AUDIO,
            data=b"fake_audio_data",
            prompt="Describe this audio"
        )
        self.assertEqual(audio_content.modality, ModalityType.AUDIO)
        self.assertEqual(audio_content.data, b"fake_audio_data")
        self.assertEqual(audio_content.prompt, "Describe this audio")
    
    def test_generation_result_creation(self):
        """Test creation of GenerationResult objects"""
        result = GenerationResult(
            text="Test response",
            cached=True,
            response_time=0.5,
            tokens_used=100
        )
        
        self.assertEqual(result.text, "Test response")
        self.assertTrue(result.cached)
        self.assertEqual(result.response_time, 0.5)
        self.assertEqual(result.tokens_used, 100)
    
    @patch('multimodal_engine.genai.Client')
    def test_get_cache_stats(self, mock_client_class):
        """Test cache statistics"""
        # Mock the Gemini client
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_client_instance.models.generate_content.return_value = mock_response
        
        engine = MultimodalEngine(api_key="test_key")
        engine.enable_caching = True
        
        # Check initial stats
        stats = engine.get_cache_stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["hit_rate"], 0)
        self.assertEqual(stats["cache_size"], 0)
        
        # Make a request
        engine.generate_from_text("Test prompt")
        
        # Check stats after first request
        stats = engine.get_cache_stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hit_rate"], 0)
        self.assertEqual(stats["cache_size"], 1)
        
        # Make the same request (should hit cache)
        engine.generate_from_text("Test prompt")
        
        # Check stats after second request
        stats = engine.get_cache_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hit_rate"], 0.5)
        self.assertEqual(stats["cache_size"], 1)
    
    @patch('multimodal_engine.genai.Client')
    def test_clear_cache(self, mock_client_class):
        """Test clearing the cache"""
        # Mock the Gemini client
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_client_instance.models.generate_content.return_value = mock_response
        
        engine = MultimodalEngine(api_key="test_key")
        engine.enable_caching = True
        
        # Make a request to populate cache
        engine.generate_from_text("Test prompt")
        
        # Check cache is populated
        stats = engine.get_cache_stats()
        self.assertEqual(stats["cache_size"], 1)
        
        # Clear cache
        engine.clear_cache()
        
        # Check cache is empty
        stats = engine.get_cache_stats()
        self.assertEqual(stats["cache_size"], 0)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)

class TestOptimizationFeatures(unittest.TestCase):
    
    def test_caching_enabled_by_default(self):
        """Test that caching is enabled by default"""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'}):
            engine = MultimodalEngine()
            self.assertTrue(engine.enable_caching)
    
    def test_compression_enabled_by_default(self):
        """Test that compression is enabled by default"""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'}):
            engine = MultimodalEngine()
            self.assertTrue(engine.enable_compression)

class TestMultimodalEngineIntegration(unittest.TestCase):
    """Integration tests for the MultimodalEngine with real API calls"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.api_key = os.environ.get('GEMINI_API_KEY')
    
    def test_api_key_must_be_set_for_integration_tests(self):
        """Test that API key must be set for integration tests"""
        if not self.api_key:
            self.fail("GEMINI_API_KEY must be set in environment to run integration tests")
    
    def test_init_with_real_api_key(self):
        """Test initialization with real API key"""
        if not self.api_key:
            self.fail("GEMINI_API_KEY must be set in environment to run integration tests")
        
        try:
            engine = MultimodalEngine()
            self.assertIsNotNone(engine)
            self.assertEqual(engine.api_key, self.api_key)
        except Exception as e:
            self.fail(f"Failed to initialize engine with real API key: {e}")
    
    def test_generate_from_text_with_real_api(self):
        """Test text generation with real API"""
        if not self.api_key:
            self.fail("GEMINI_API_KEY must be set in environment to run integration tests")
        
        try:
            engine = MultimodalEngine()
            result = engine.generate_from_text("What is the capital of France?")
            
            self.assertIsInstance(result, GenerationResult)
            self.assertIsInstance(result.text, str)
            self.assertGreater(len(result.text), 0)
            self.assertFalse(result.cached)  # First call shouldn't be cached
            self.assertGreater(result.response_time, 0)
        except Exception as e:
            self.fail(f"Failed to generate text with real API: {e}")
    
    def test_generate_from_text_with_caching_real_api(self):
        """Test caching with real API"""
        if not self.api_key:
            self.fail("GEMINI_API_KEY must be set in environment to run integration tests")
        
        try:
            engine = MultimodalEngine()
            engine.enable_caching = True
            
            # First call
            result1 = engine.generate_from_text("Explain what caching is.")
            self.assertFalse(result1.cached)
            
            # Second call (should be cached)
            result2 = engine.generate_from_text("Explain what caching is.")
            self.assertTrue(result2.cached)
            self.assertEqual(result1.text, result2.text)
        except Exception as e:
            self.fail(f"Failed to test caching with real API: {e}")

def run_tests():
    """Run all tests and provide detailed results"""
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMultimodalEngineStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestMultimodalEngineUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizationFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestMultimodalEngineIntegration))
    
    # Create a custom test result class to capture all results
    class CustomTestResult(unittest.TextTestResult):
        def __init__(self, stream, descriptions, verbosity):
            super().__init__(stream, descriptions, verbosity)
            self.test_results = []
            
        def addSuccess(self, test):
            super().addSuccess(test)
            self.test_results.append({
                'name': test._testMethodName,
                'status': 'PASS',
                'details': ''
            })
            
        def addError(self, test, err):
            super().addError(test, err)
            self.test_results.append({
                'name': test._testMethodName,
                'status': 'ERROR',
                'details': self._exc_info_to_string(err, test)
            })
            
        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.test_results.append({
                'name': test._testMethodName,
                'status': 'FAIL',
                'details': self._exc_info_to_string(err, test)
            })
            
        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            self.test_results.append({
                'name': test._testMethodName,
                'status': 'SKIP',
                'details': reason
            })
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2, resultclass=CustomTestResult)
    result = runner.run(suite)
    
    # Print detailed summary
    print("\n" + "="*60)
    print("MULTIMODAL ENGINE TEST RESULTS")
    print("="*60)
    
    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    
    for test_result in result.test_results:
        status = test_result['status']
        if status == 'PASS':
            passed += 1
        elif status == 'FAIL':
            failed += 1
        elif status == 'ERROR':
            errors += 1
        elif status == 'SKIP':
            skipped += 1
            
        print(f"\n{status}: {test_result['name']}")
        if status in ['FAIL', 'ERROR'] and test_result['details']:
            print(f"  Details: {test_result['details'][:200]}...")
        elif status == 'SKIP':
            print(f"  Reason: {test_result['details']}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tests run: {len(result.test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    
    # Show API key status
    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        print(f"\nAPI Key: SET (ends with ...{api_key[-4:]})")
        print("Integration tests with real API will run.")
    else:
        print("\nAPI Key: NOT SET")
        print("Integration tests with real API will FAIL.")
    
    if failed > 0 or errors > 0:
        print("\nFAILED TESTS:")
        for test_result in result.test_results:
            if test_result['status'] in ['FAIL', 'ERROR']:
                print(f"  - {test_result['name']} ({test_result['status']})")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Run all tests and provide detailed results
    success = run_tests()
    
    if success:
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED")
        print("="*60)
        sys.exit(1)
# Multimodal Engine Library

A Python library for interacting with Google's Gemini model using multiple modalities (text, image, audio) without any web framework dependencies.

## Features

- Text generation from text prompts
- Content analysis from images
- Content analysis from audio files (MP3 format)
- Combined multimodal analysis (text + image + audio)
- Built-in caching for optimization
- Image compression for faster processing
- Comprehensive error handling

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your Gemini API key in the `.env` file:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## Usage

### Basic Usage

```python
from multimodal_engine import MultimodalEngine

# Initialize the engine
engine = MultimodalEngine()

# Generate text from prompt
result = engine.generate_from_text("Write a short poem about technology")
print(result.text)

# Analyze an image
result = engine.generate_from_image("path/to/image.jpg", "Describe this image in detail")
print(result.text)

# Analyze an audio file
result = engine.generate_from_audio("path/to/audio.mp3", "Describe this audio clip")
print(result.text)
```

### Multimodal Analysis

```python
from multimodal_engine import MultimodalEngine, MultimodalContent, ModalityType

# Initialize the engine
engine = MultimodalEngine()

# Create multimodal content
contents = [
    MultimodalContent(
        modality=ModalityType.TEXT,
        data="This is a technical document about machine learning"
    ),
    MultimodalContent(
        modality=ModalityType.IMAGE,
        data="path/to/chart.png"
    )
]

# Analyze all modalities together
result = engine.generate_from_multimodal(contents, "Analyze these inputs and explain how they relate to each other")
print(result.text)
```

### Optimization Features

```python
from multimodal_engine import MultimodalEngine

# Initialize the engine
engine = MultimodalEngine()

# Enable/disable caching
engine.enable_caching = True  # Default is True

# Enable/disable image compression
engine.enable_compression = True  # Default is True

# Check cache statistics
stats = engine.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2f}")

# Clear cache
engine.clear_cache()
```

## Supported File Formats

### Images
- JPEG - image/jpeg
- PNG - image/png
- WEBP - image/webp
- HEIC - image/heic
- HEIF - image/heif

### Audio
- MP3 - audio/mp3 (primary support)
- WAV - audio/wav
- AIFF - audio/aiff
- AAC - audio/aac
- OGG Vorbis - audio/ogg
- FLAC - audio/flac

## API Reference

### MultimodalEngine

#### `__init__(self, api_key=None, model_name="gemini-2.5-flash")`
Initialize the multimodal engine.

**Parameters:**
- `api_key` (str, optional): Google Gemini API key
- `model_name` (str): Name of the Gemini model to use

#### `generate_from_text(self, prompt, stream=False)`
Generate content from text prompt.

**Parameters:**
- `prompt` (str): Text prompt for generation
- `stream` (bool): Whether to stream the response

**Returns:**
- `GenerationResult`: Result with generated text

#### `generate_from_image(self, image_data, prompt="Describe this image.")`
Generate content from image.

**Parameters:**
- `image_data` (bytes or str): Image bytes or file path
- `prompt` (str): Prompt for image analysis

**Returns:**
- `GenerationResult`: Result with generated text

#### `generate_from_audio(self, audio_data, prompt="Describe this audio clip.")`
Generate content from audio.

**Parameters:**
- `audio_data` (bytes or str): Audio bytes or file path
- `prompt` (str): Prompt for audio analysis

**Returns:**
- `GenerationResult`: Result with generated text

#### `generate_from_multimodal(self, contents, prompt=None)`
Generate content from multiple modalities.

**Parameters:**
- `contents` (List[MultimodalContent]): List of content objects
- `prompt` (str, optional): General prompt for all modalities

**Returns:**
- `GenerationResult`: Result with generated text

### Data Classes

#### `MultimodalContent`
Represents content for multimodal processing.

**Attributes:**
- `modality` (ModalityType): Type of content (TEXT, IMAGE, AUDIO)
- `data` (Any): Content data (text, bytes, or file path)
- `prompt` (str, optional): Specific prompt for this content
- `mime_type` (str, optional): MIME type for the content

#### `GenerationResult`
Result of content generation.

**Attributes:**
- `text` (str): Generated text content
- `cached` (bool): Whether the result was retrieved from cache
- `response_time` (float): Time taken to generate the response
- `tokens_used` (int, optional): Number of tokens used

### Enums

#### `ModalityType`
Enumeration of supported modalities.
- `TEXT`
- `IMAGE`
- `AUDIO`
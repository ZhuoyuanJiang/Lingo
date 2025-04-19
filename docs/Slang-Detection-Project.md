# Slang Detection Project Documentation

## Project Overview
This project creates a pipeline to detect and explain slang terms from video content:
1. Downloads audio from YouTube videos
2. Transcribes audio to text
3. Detects slang terms in the transcribed text
4. Generates example sentences for detected slang terms

## Dependencies
- `yt_dlp`: YouTube video downloader (requires FFmpeg)
- `whisper`: OpenAI's speech recognition model
- `pandas`: Data manipulation
- `wordfreq`: Word frequency statistics
- `spacy`: Natural language processing (requires `en_core_web_sm` model)
- `transformers`: Text generation with GPT-2

## Key Functions

### `download_audio(url, output_path='audio.mp3')`
Downloads audio from a YouTube video.
```python
# Example
download_audio("https://www.youtube.com/watch?v=7fMKxYBNCfc")
```

### `extract_slang_phrases(text)`
Identifies slang terms in text.
```python
# Example
text = "That party was lit!"
slangs = extract_slang_phrases(text)
print(slangs)  # ['lit']
```

### `generate_example(slang, definition)`
Creates an example sentence for a slang term.
```python
# Example
example = generate_example("ghosted", "When someone cuts off all communication")
```

## Data Processing
The project processes Urban Dictionary data with these filters:
- Removes entries with missing words/definitions
- Sorts by community score (upvotes - downvotes)
- Filters out common words (Zipf frequency ≥ 4)
- Filters out very short words (≤ 3 characters)

## Future Improvements
As noted in the code:
- Build a better slang detection model using BERT or similar
- Improve the quality of generated examples (current GPT-2 output is suboptimal)
- Implement real-time conversion for web interfaces
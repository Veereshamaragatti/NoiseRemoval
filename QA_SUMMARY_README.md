# 🤖 AI Video Q&A & Summary Feature

## Overview

This feature uses **Groq's Llama 3.3 70B** AI model to provide intelligent Q&A and comprehensive video summaries based on your video subtitles.

## Features

### 📋 Video Summary
- **Overall Summary**: 3-5 sentence comprehensive overview
- **Key Points**: 6 main takeaways from the video
- **Minute-by-Minute Breakdown**: Detailed timeline of content

### 💬 Interactive Q&A
- Ask any question about video content
- Timestamp-based queries (e.g., "What did they say at minute 5?")
- Context-aware responses with relevant quotes
- Chat history maintained during session

## How to Use

### 1. Process a Video
First, upload and process a video with transcription enabled:
- Upload video file
- Enable "📝 Enable Transcription"
- Select desired languages
- Click "Upload & Process Video"

### 2. Access Q&A & Summary
After processing completes:
- Click "🤖 Q&A & Summary" button in video controls
- The Q&A and Summary panels will appear below the video

### 3. Generate Summary
- Click "✨ Generate" in the Summary panel
- AI will analyze the subtitles and create:
  - Overall summary
  - Key points (numbered)
  - Minute-by-minute breakdown

### 4. Ask Questions
In the Q&A panel:
- Type your question in the input box
- Press Enter or click "📤 Send"
- AI will respond based on video content

#### Example Questions:
- "What is the main topic of this video?"
- "What did they say about AI at minute 5?"
- "Summarize what happens at 10:30"
- "What are the key conclusions?"
- "Explain the concept mentioned at 3:15"

## API Endpoints

### Generate Summary
```http
POST /api/summary/{video_id}?lang=en
```

**Response:**
```json
{
  "status": "success",
  "video_id": "2db11364",
  "language": "en",
  "data": {
    "summary": "Overall summary text...",
    "minuteByMinute": [
      "Minute 0-1: Introduction to topic",
      "Minute 1-2: Main concept explained"
    ],
    "keyPoints": [
      "Point 1",
      "Point 2"
    ]
  }
}
```

### Ask Question
```http
POST /api/ask
Content-Type: multipart/form-data

video_id=2db11364
question=What is the main topic?
lang=en
```

**Response:**
```json
{
  "status": "success",
  "question": "What is the main topic?",
  "answer": "Based on the video subtitles, the main topic is...",
  "video_id": "2db11364",
  "language": "en"
}
```

### Generate Assessment (Optional)
```http
POST /api/assessment/{video_id}?lang=en
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "mcqs": [
      {
        "question": "What is...?",
        "options": ["A", "B", "C", "D"],
        "correct": 0,
        "explanation": "Explanation..."
      }
    ],
    "shortQuestions": [
      {
        "question": "Explain...",
        "answer": "Model answer..."
      }
    ]
  }
}
```

## Configuration

### Groq API Key
The API key is configured in `backend/groq_api.py`:

```python
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-api-key-here")
```

**For production**, set it as an environment variable:
```bash
export GROQ_API_KEY="your-api-key-here"
```

### Model Settings
- **Model**: llama-3.3-70b-versatile
- **Temperature**: 0.7 (balanced creativity/accuracy)
- **Max Tokens**: 2000 (sufficient for detailed responses)

## UI Design

The interface uses a **black and white theme** for a professional, focused experience:

### Color Scheme
- **Background**: Linear gradient (#1a1a1a to #2d2d2d)
- **Cards**: Dark with subtle borders (#444)
- **Buttons**: White gradient with black text
- **Messages**: 
  - User: White background
  - Assistant: Dark background
- **Accents**: Gray tones (#666, #999)

### Features
- **Responsive Design**: Works on desktop and mobile
- **Smooth Animations**: Fade-in effects for messages
- **Hover Effects**: Cards lift on hover
- **Scrollable Chat**: Auto-scroll to latest message
- **Loading Indicators**: Spinners for AI processing

## Technical Details

### Backend Components
1. **groq_api.py**: Groq API integration
   - `generate_summary()`: Creates video summary
   - `answer_question()`: Handles Q&A
   - `generate_assessment()`: Creates MCQs (optional)
   - `evaluate_short_answer()`: Grades answers (optional)

2. **app.py**: FastAPI endpoints
   - `/api/summary/{video_id}`: Summary generation
   - `/api/ask`: Question answering
   - `/api/assessment/{video_id}`: Assessment creation
   - `/api/evaluate`: Answer evaluation

3. **vtt_utils.py**: Subtitle processing
   - `load_vtt_text()`: Extracts text from VTT files

### Frontend Components
1. **index_advanced.html**: Main interface
   - Q&A chat interface
   - Summary display with sections
   - Toggle controls
   - API integration

## Requirements

### Python Packages
```bash
requests>=2.31.0  # For Groq API calls
```

### Subtitle Files
- Video must be processed with transcription enabled
- VTT files must exist in `backend/subtitles/` directory
- Format: `{video_id}.{lang}.vtt`

## Troubleshooting

### "Subtitle file not found" Error
- Ensure transcription was enabled during video processing
- Check that VTT files exist in `backend/subtitles/` directory
- Verify the video ID is correct

### "Groq API error" Message
- Check your API key is valid
- Verify internet connection
- Ensure API rate limits not exceeded

### Summary Not Generating
- Wait for video processing to complete
- Make sure backend server is running on port 8001
- Check browser console for errors (F12)

### Questions Not Answered
- Ensure the question is related to video content
- Try rephrasing the question
- Check that the correct language is selected

## Best Practices

### For Better Summaries
1. Use clear, well-transcribed audio
2. Process longer videos (more context)
3. Select appropriate language
4. Wait for complete processing

### For Better Q&A
1. Ask specific questions
2. Reference timestamps when needed
3. Use proper grammar
4. One question at a time

### Performance Tips
1. Generate summary once, refer to it
2. Clear chat when starting new topic
3. Use keyboard shortcuts (Enter to send)
4. Keep questions concise

## Example Workflow

```
1. Upload video → Select English + Hindi
2. Process with transcription enabled
3. Wait for completion (~5-10 minutes)
4. Click "🤖 Q&A & Summary"
5. Generate summary first (understand content)
6. Ask follow-up questions as needed
7. Download summary if needed (future feature)
```

## Future Enhancements

- [ ] PDF export for summaries
- [ ] PDF export for Q&A conversations
- [ ] Voice input for questions
- [ ] Multi-language Q&A (ask in any language)
- [ ] Save/load chat history
- [ ] Share summaries/Q&A
- [ ] Quiz generation and grading
- [ ] Flashcard creation
- [ ] Study notes generation

## Credits

- **AI Model**: Groq Llama 3.3 70B
- **API**: Groq Cloud API
- **Framework**: FastAPI + Vanilla JavaScript
- **Design**: Custom black & white theme

## Support

For issues or questions:
1. Check this documentation
2. Review browser console for errors
3. Verify backend logs
4. Ensure all dependencies installed
5. Check Groq API status

---

**Built with ❤️ for educational content processing**

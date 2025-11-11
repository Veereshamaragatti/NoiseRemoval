"""
Groq API integration for video summary and Q&A
Uses Llama 3.3 70B model for intelligent content analysis
"""

import os
import json
import requests
from typing import Dict, List, Optional

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it before running the application.")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"


def call_groq(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """
    Call Groq API with chat messages
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in response
        
    Returns:
        API response content as string
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Groq API error: {str(e)}")


def clean_json_response(response: str) -> str:
    """Remove markdown code blocks from JSON response"""
    cleaned = response.strip()
    
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    return cleaned.strip()


def generate_summary(subtitles: str) -> Dict:
    """
    Generate comprehensive video summary
    
    Args:
        subtitles: Full subtitle text from VTT file
        
    Returns:
        Dictionary with summary, minuteByMinute breakdown, and keyPoints
    """
    prompt = f"""Analyze the following video subtitles and provide:
1. An overall summary of the video (3-5 sentences) that is well-formatted and suitable for studying
2. A minute-by-minute breakdown as an array of strings (each entry like "Minute 0-1: summary content")
3. 6 key points from the entire content

Subtitles:
{subtitles}

Respond in JSON format:
{{
  "summary": "A comprehensive overview of the video content in 3-5 well-structured sentences.",
  "minuteByMinute": ["Minute 0-1: content", "Minute 1-2: content", "Minute 2-3: content", ...],
  "keyPoints": ["point 1", "point 2", ...]
}}"""

    messages = [
        {
            "role": "system",
            "content": "You are an educational content analyzer. Always respond with valid JSON containing summary, minuteByMinute array, and keyPoints array."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    response = call_groq(messages, temperature=0.7, max_tokens=2000)
    
    try:
        cleaned = clean_json_response(response)
        result = json.loads(cleaned)
        
        # Validate structure
        if not all(key in result for key in ["summary", "minuteByMinute", "keyPoints"]):
            raise ValueError("Missing required keys in response")
        
        return result
    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse summary response: {e}")
        return {
            "summary": response,
            "minuteByMinute": [],
            "keyPoints": ["Unable to parse key points"]
        }


def answer_question(subtitles: str, question: str) -> str:
    """
    Answer questions about video content based on subtitles
    
    Args:
        subtitles: Full subtitle text from VTT file
        question: User's question about the video
        
    Returns:
        Answer as markdown-formatted string
    """
    prompt = f"""Based on the following video subtitles, answer the user's question. 

IMPORTANT: 
- If the user asks about a specific time/minute (e.g., "What did they say at minute 5?"), look for content around that timestamp and provide a detailed summary of what was discussed.
- If the user asks about a topic with a time reference (e.g., "What did they say about AI at minute 10?"), find the relevant section and provide context.
- If the answer is not in the subtitles, respond with "This information was not mentioned in the video."
- Always be specific and quote relevant parts when available.

Subtitles:
{subtitles}

Question: {question}"""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on video transcripts. You can understand timestamp queries and retrieve specific moments from the transcript."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    return call_groq(messages, temperature=0.7, max_tokens=2000)


def generate_assessment(subtitles: str) -> Dict:
    """
    Generate MCQs and short answer questions from video content
    
    Args:
        subtitles: Full subtitle text from VTT file
        
    Returns:
        Dictionary with mcqs and shortQuestions arrays
    """
    prompt = f"""Based on the following video subtitles, create an assessment:

1. Generate 5 multiple choice questions (MCQs) with 4 options each
2. Generate 3 short answer questions with model answers

IMPORTANT: For each MCQ, provide an explanation that describes why the correct answer is right and why the other options are incorrect.

Subtitles:
{subtitles}

Respond in JSON format:
{{
  "mcqs": [
    {{
      "question": "question text",
      "options": ["option1", "option2", "option3", "option4"],
      "correct": 0,
      "explanation": "Detailed explanation of why this answer is correct and why others are wrong"
    }}
  ],
  "shortQuestions": [
    {{
      "question": "question text",
      "answer": "model answer"
    }}
  ]
}}"""

    messages = [
        {
            "role": "system",
            "content": "You are an educational assessment creator. Always respond with valid JSON."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    response = call_groq(messages, temperature=0.7, max_tokens=2000)
    
    try:
        cleaned = clean_json_response(response)
        result = json.loads(cleaned)
        return result
    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse assessment response: {e}")
        return {
            "mcqs": [],
            "shortQuestions": []
        }


def evaluate_short_answer(question: str, model_answer: str, user_answer: str) -> Dict:
    """
    Evaluate student's short answer against model answer
    
    Args:
        question: The question text
        model_answer: Expected/model answer
        user_answer: Student's submitted answer
        
    Returns:
        Dictionary with score (0-2) and feedback
    """
    prompt = f"""Evaluate the following answer and provide:
1. A score out of 2:
   - 2 marks: Correct and complete
   - 1.5 marks: Mostly correct with minor issues
   - 1 mark: Partially correct
   - 0.5 marks: Minimal understanding
   - 0 marks: Incorrect or no answer

2. Detailed feedback explaining:
   - What was good about the answer
   - What was missing or incorrect
   - How it compares to the model answer

Question: {question}
Model Answer: {model_answer}
User Answer: {user_answer}

Respond in JSON format:
{{
  "score": 1.5,
  "feedback": "Your detailed feedback here"
}}"""

    messages = [
        {
            "role": "system",
            "content": "You are an assessment grader. Always respond with valid JSON containing score and feedback."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    response = call_groq(messages, temperature=0.5, max_tokens=1000)
    
    try:
        cleaned = clean_json_response(response)
        result = json.loads(cleaned)
        
        # Clamp score between 0 and 2
        score = max(0, min(2, result.get("score", 0)))
        feedback = result.get("feedback", "No feedback available")
        
        return {"score": score, "feedback": feedback}
    
    except (json.JSONDecodeError, ValueError):
        return {"score": 0, "feedback": "Error evaluating answer"}

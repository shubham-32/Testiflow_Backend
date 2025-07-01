from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
import logging
import os
from dotenv import load_dotenv
import asyncio
from bs4 import BeautifulSoup
import requests
import re


# Load environment variables from .env file
load_dotenv()

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins, especially localhost for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://testiflow.onrender.com"],  # You can restrict it to only your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for input validation
class ReviewText(BaseModel):
    questionId: str
    answer: str

class Review(BaseModel):
    name: str
    email: str
    reviewText: List[ReviewText]
    isFav: bool = False
    rating: Optional[int] = None

class Question(BaseModel):
    id: str
    question: str

class SummarizationRequest(BaseModel):
    questions: List[Question]
    reviews: List[Review]

class CaseStudyRequest(BaseModel):
    questions: List[Question]
    reviews: List[Review]

class WebsiteRequest(BaseModel):
    url: str

# Function to map questions and answers
def map_questions_and_answers(questions, review_text):
    mapped_reviews = []
    
    for i, question in enumerate(questions):
        # Match the question and answer, if available
        answer = review_text[i].answer if i < len(review_text) else "No answer provided"
        mapped_reviews.append({"question": question.question, "answer": answer})
    
    return mapped_reviews


# Function to generate a summary
def generate_summary(questions_and_reviews):
    context = "Summarize the following testimonial according to the questions provided in 1 paragraph (short and brief) and highlight the important details (positive with green text and negative with red text):\n"
    for qa in questions_and_reviews:
        context += f"Questions: {qa['question']}\nAnswer: {qa['answer']}\n"
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logging.error("GROQ_API_KEY not set in environment variables.")
        raise ValueError("GROQ_API_KEY not set in environment variables.")
    
    client = Groq(api_key=api_key)  # Pass the API key to the Groq client
    completion = client.chat.completions.create(  # Use sync Groq call instead of async
        model="llama-3.1-8b-instant", 
        messages=[{"role": "user", "content": context}],
        temperature=0.7,
        max_tokens=200,
        top_p=1,
        stream=False,
        stop=None,
    )
    
    logging.info(f"Groq response: {completion}")

    try:
        # Correct access to message content
        message_content = completion.choices[0].message.content
    except AttributeError as e:
        logging.error(f"Error accessing the message content: {str(e)}")
        raise HTTPException(status_code=500, detail="Error accessing the response content.")
    
    return message_content.strip()


def generate_casestudy(questions_and_reviews):
    context = "This data represents reviews from different users regarding some questions, so make a case study to better understand overall sentiment in brief."
    
    for qa in questions_and_reviews:
        context += f"Questions: {qa['question']}\nAnswers:\n"
        for answer in qa['answers']:
            context += f"- {answer}\n"
        context += "\n"
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logging.error("GROQ_API_KEY not set in environment variables.")
        raise ValueError("GROQ_API_KEY not set in environment variables.")
    
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(  # Use sync Groq call instead of async
        model="llama-3.1-8b-instant", 
        messages=[{"role": "user", "content": context}],
        temperature=0.7,
        max_tokens=1000,
        top_p=1,
        stream=False,
        stop=None,
    )
    
    logging.info(f"Groq response: {completion}")

    try:
        # Correct access to message content
        message_content = completion.choices[0].message.content
    except AttributeError as e:
        logging.error(f"Error accessing the message content: {str(e)}")
        raise HTTPException(status_code=500, detail="Error accessing the response content.")
    
    return message_content.strip()

def extract_website_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract relevant content (for simplicity, using the title and description)
        title = soup.title.string if soup.title else 'No title'
        meta_description = soup.find('meta', attrs={'name': 'description'})
        description = meta_description['content'] if meta_description else 'No description'
        
        # Return extracted content
        return {'title': title, 'description': description}
    
    except Exception as e:
        logging.error(f"Error extracting website content: {str(e)}")
        return {'error': str(e)}

# Function to generate testimonial questions based on website content
def generate_custom_message(website_content):
    if website_content['title'] != 'No title':
        return f"We appreciate your feedback on {website_content['title']}. Your input helps us improve!"
    return "We value your feedback! Let us know how we can enhance your experience on this website."

def generate_header_title(website_content):
    # Simple header title logic based on the website title
    if website_content['title'] != 'No title':
        return f"Feedback for {website_content['title']}"
    return "Feedback for the Website"

def generate_testimonial_questions(website_content):
    # Prepare the context for question generation
    context = f"Generate 5 questions based on the following website information in order to get user feedback about the website:\n"
    context += f"Title: {website_content['title']}\n"
    context += f"Description: {website_content['description']}\n"
    
    # Get the API key from environment variables
    api_key = os.getenv("GROQ_API_KEY")
    
    # Check if the API key is set
    if not api_key:
        logging.error("GROQ_API_KEY not set in environment variables.")
        raise ValueError("GROQ_API_KEY not set in environment variables.")
    
    # Use Groq to generate the testimonial questions
    client = Groq(api_key=api_key)  # Pass the API key to the Groq client
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Use the appropriate model
            messages=[{"role": "user", "content": context}],
            temperature=0.7,
            max_tokens=150,  # Adjust based on the desired response length
            top_p=1,
            stream=False,
            stop=None,
        )
        
        # Inspect the completion object
        logging.info(f"Groq response: {completion}")
        
        # Access the response and clean up the generated questions
        message_content = completion.choices[0].message.content.strip()
        
        # Use regex to capture all questions that start with a number (1., 2., 3., etc.) and end with a newline
        questions = re.findall(r'\d+\.\s[^\n]+', message_content)
        
        # Return the list of questions
        return questions

    except Exception as e:
        logging.error(f"Error generating testimonial questions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate testimonial questions.")


# FastAPI route to generate case study
@app.post("/generate-casestudy")
def generate_casestudy_route(request: CaseStudyRequest):
    try:
        # logging.info(f"Received request with questions: {request.questions} and reviews: {request.reviews}")
        
        questions_and_answers = []
        
        for question in request.questions:
            answers = []
            for review in request.reviews:
                for review_text in review.reviewText:
                    if review_text.questionId == question.id:
                        answers.append(review_text.answer)
            
            questions_and_answers.append({
                "question": question.question,
                "answers": answers
            })
        print(questions_and_answers)
        summary = generate_casestudy(questions_and_answers)  # Call sync function
        return JSONResponse(content={"summary": summary})
    
    except Exception as e:
        logging.error(f"Error generating case study: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating case study.")


# Endpoint to summarize reviews
@app.post("/summarize-reviews")
def summarize_reviews(request: SummarizationRequest):
    try:
        all_reviews_text = []
        for review in request.reviews:
            mapped_reviews = map_questions_and_answers(request.questions, review.reviewText)
            all_reviews_text.extend(mapped_reviews)
        
        summary = generate_summary(all_reviews_text)  # Call sync function
        return JSONResponse(content={"summary": summary})
    
    except Exception as e:
        logging.error(f"Error during summarization: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate summary.")


@app.post("/generate-testimonial-questions")
async def generate_testimonial_questions_endpoint(request: WebsiteRequest):
    try:
        # Extract website content using the provided URL
        website_content = extract_website_content(request.url)
        
        # If there was an error extracting content, return an error response
        if 'error' in website_content:
            raise HTTPException(status_code=400, detail=website_content['error'])
        
        # Generate testimonial questions based on extracted content
        questions = generate_testimonial_questions(website_content)
        header = generate_header_title(website_content)
        message = generate_custom_message(website_content)
        
        # Return the generated questions as a response
        return {"questions": questions, "header": header, "message": message}
    
    except Exception as e:
        logging.error(f"Error during the process: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate testimonial questions.")

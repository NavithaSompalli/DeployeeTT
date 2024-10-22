from fastapi import APIRouter, FastAPI, BackgroundTasks, Request, HTTPException, Depends
import openai
import asyncio
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from celery import Celery
import redis
import json
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
import uuid
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

# Set up Celery to manage background tasks for scalability
celery_app = Celery(__name__, broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# Set up Redis connection pool to manage session data efficiently
pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
redis_client = redis.Redis(connection_pool=pool)

# Set OpenAI API key from environment variable for better security
api_key = "your open api key"
openai.api_key = api_key

# Initialize FastAPI app
app = FastAPI()

# Set up rate limiter to control the number of requests per user
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Allow requests from specified origins with CORS middleware for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your actual front-end domain
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Set up logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Pydantic models for input data
class PromptText(BaseModel):
    prompt_text: str

class ResumeUpload(BaseModel):
    resume_content: str

# Set up API router with a prefix
router = APIRouter(prefix="/api")

# Endpoint to upload resume and generate a human-like response
@router.post("/upload_resume", tags=["Resume"])
async def upload_resume(request: Request, resume: ResumeUpload):
    try:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        logger.info(f"Generated new session_id: {session_id}")

        # Extract resume content
        resume_content = resume.resume_content

        # Store resume content in Redis for future reference
        redis_client.set(f"resume_{session_id}", json.dumps({
            "content": resume_content
        }), ex=86400)  # Set an expiration of 24 hours for large data handling
        return {"message": "Resume uploaded successfully", "session_id": session_id}
    except Exception as error:
        logger.error(f"Error uploading resume: {error}")
        return {"error": f"Error uploading resume: {error}"}

# Endpoint to generate a human-like response
@router.post("/get_answer", tags=["Response"])
@limiter.limit("10/minute")
async def generate_humanlike_response(request: Request, request_body: PromptText, session_id: str):
    try:
        logger.info(f"Processing request for session_id: {session_id}")

        # Retrieve resume content from Redis if it exists
        resume_data = redis_client.get(f"resume_{session_id}")
        if resume_data:
            resume_data = json.loads(resume_data)
            resume_summary = resume_data.get("content", "No resume content provided.")
        else:
            resume_summary = "No resume content provided."

        # Retrieve conversation history from Redis if it exists, otherwise initialize a new one
        if redis_client.exists(session_id):
            logger.info(f"Retrieving conversation history for session_id: {session_id}")
            conversation_history = json.loads(redis_client.get(session_id))
        else:
            logger.info(f"No conversation history found for session_id: {session_id}, initializing new history")
            conversation_history = []

        # Append the user's input to the conversation history
        conversation_history.append({"role": "user", "content": request_body.prompt_text})
        logger.info(f"User input added to conversation history for session_id: {session_id}")

        # Limit the conversation history to the last 10 interactions for context
        limited_history = conversation_history[-10:]

        # Create a detailed prompt based on the user's input and resume content
        prompt = f"""You're an experienced professional based on this resume content:
        {resume_summary}
        I'm the recruiter asking questions to assess your skills and expertise. Respond to any technical or non-technical question in normal Indian slang English in a concise manner, and conversational manner, as if we're speaking face-to-face, showcasing your achievements, skills, and problem-solving abilities based on the question. Respond without using any large and complicated vocabulary, complicated words and complicated terminologies in the speech, and give the answer short and concise without dramatic words.
        Answer my questions as you are a human: {request_body.prompt_text}"""

        # Construct the messages for OpenAI API with system message and user history
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + limited_history
        messages.append({"role": "user", "content": prompt})
        logger.info(f"Constructed messages for OpenAI API for session_id: {session_id}")

        # Call OpenAI API directly to get the response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=2048  # Ensure enough token space for large responses
        )
        response_content = response['choices'][0]['message']['content'].strip()

        # Append the assistant's response to the conversation history
        conversation_history.append({"role": "assistant", "content": response_content})
        # Update the conversation history in Redis
        redis_client.set(session_id, json.dumps(conversation_history))

        logger.info(f"Response generated successfully for session_id: {session_id}")
        return {"response": response_content}

    except Exception as error:
        logger.error(f"Error with OpenAI API for session_id: {session_id}: {error}")
        return {"error": f"Error with OpenAI API: {error}"}

# Endpoint to check the health of the Redis connection
@router.get("/health", tags=["Health"])
async def health_check():
    try:
        logger.info("Performing health check for Redis connection")
        redis_client.ping()
        logger.info("Redis connection is healthy")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

# Include the router in the FastAPI app
app.include_router(router)

# Authentication Setup
SECRET_KEY = "your_secret_key"  # Replace with a secure random key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Define the password hashing scheme
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 password bearer for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# In-memory user store for demonstration purposes (replace with a real database)
fake_users_db = {
    "test@example.com": {
        "full_name": "Test User",
        "email": "test@example.com",
        "hashed_password": pwd_context.hash("password123"),
        "disabled": False,
    }
}

# Function to verify the user's credentials
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Function to get the user from the fake database
def get_user(db, email: str):
    if email in db:
        user_dict = db[email]
        return user_dict

# Function to create a new access token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Login endpoint for user authentication
@app.post("/token", tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(fake_users_db, form_data.username)
    if not user or not verify_password(form_data.password, user['hashed_password']):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['email']}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Dependency to get the current user from the token
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, email)
    if user is None:
        raise credentials_exception
    return user

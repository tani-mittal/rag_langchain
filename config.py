import os
from pathlib import Path

RAW_DATA_PATH = Path("")
)

# AWS Configuration
AWS_REGION = "u"
region = ''
model_id = ""
model_arn = ""
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"
BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# AWS Credentials - Use None to let boto3 use default credential chain
# This will work with Okta, AWS profiles, IAM roles, etc.
AWS_ACCESS_KEY_ID = None
AWS_SECRET_ACCESS_KEY = None
AWS_SESSION_TOKEN = None  # Important for temporary credentials from Okta

# S3 Configuration
S3_BUCKET = ""
S3_PREFIX = ""

# Retrieval Configuration
RETRIEVAL_K = 5

# Processing Configuration
CHUNK_SIZES = {
    'pdf': 1200,
    'docx': 1000,
    'pptx': 800,
    'xlsx': 600
}

CHUNK_OVERLAPS = {
    'pdf': 200,
    'docx': 150,
    'pptx': 100,
    'xlsx': 50
}

# Memory Configuration
MEMORY_WINDOW_SIZE = 10
MAX_CHAT_HISTORY = 100

# Accuracy Tracking
CONFIDENCE_THRESHOLD = 0.7
MIN_RETRIEVAL_SCORE = 0.5
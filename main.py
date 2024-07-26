import io
import os
import logging
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for API Keys
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("Google API key is not set.")
    raise ValueError("Google API key is not set.")

if not os.getenv("COHERE_API_KEY"):
    logger.error("Cohere API key is not set.")
    raise ValueError("Cohere API key is not set.")

# Configure Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Embedding API Key
cohere_api_key = os.getenv("COHERE_API_KEY")

# Function to extract text from PDF
def get_pdf_text(pdf: bytes) -> str:
    text = ""
    with io.BytesIO(pdf) as pdf_buffer:
        pdf_reader = PdfReader(pdf_buffer)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=350)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def create_vector_store(text_chunks):
    cohere_embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-light-v3.0")
    vector_store = FAISS.from_texts(text_chunks, cohere_embeddings)
    return vector_store

# Function to get conversational chain
def get_conversational_chain():
    prompt_template = """
        **Objective:**

        Create a comprehensive lesson plan based on the content of the provided text document. The lesson plan should be suitable for the specified grade level and tailored to the core content of the document.

        **Instructions:**

        1. **Analyze the Document:**
            - Thoroughly review the entire document to understand its main ideas and key information.
            - Identify the primary themes and important details relevant to the subject matter.

        2. **Lesson Plan Structure:**
            - **Title:** Include a concise title for the lesson.
            - **Grade Level:** Specify the grade level for which the lesson is intended.
            - **Objectives:** Define clear learning objectives that align with the core content of the document.
            - **Materials Needed:** List any materials or resources required for the lesson.
            - **Introduction:** Provide an engaging introduction to the topic.
            - **Activities:** Outline activities and exercises that will help students grasp the content. Include step-by-step instructions.
            - **Assessment:** Describe how students' understanding will be evaluated.
            - **Conclusion:** Summarize the lesson and suggest follow-up activities or additional resources.

        3. **Formatting:**
            - Use clear headings, bullet points, and lists to enhance readability and organization.
            - Ensure the lesson plan is easy to follow and understand.

        **Context:**
        {context}

        **Question:**
        {question}
        """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to generate lesson plan
def generate_lesson_plan(context, user_query):
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": context, "question": user_query}, return_only_outputs=True)
    return response["output_text"]

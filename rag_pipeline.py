import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Required to prevent warnings in Transformers v4.10+
import random
import string
import re
from fpdf import FPDF
import fitz  # PyMuPDF
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file
import uuid  # Add this import at the top of your file


# NEW imports for the recommended MongoClient usage
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import openai


########################################################################
# STEP A: Generate Dummy Resumes as PDF Files
########################################################################

def random_person_name():
    """Generate a random 'FirstName LastName'."""
    first_names = ["John", "Jane", "Alex", "Emily", "Chris", "Samantha", "Michael", "Sophia", "Ethan", "Olivia"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Taylor", "Martinez", "Anderson", "Lopez", "Harris", "Clark"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def random_phone():
    """Generate a random phone number (US-style)."""
    return f"+1-{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"

def random_email(name):
    """Generate a random email based on the name."""
    handle = name.lower().replace(" ", ".")
    domains = ["gmail.com", "yahoo.com", "outlook.com", "example.com"]
    return f"{handle}@{random.choice(domains)}"

def random_skills(job):
    """Generate job-specific skills to add more variety."""
    skill_map = {
        "plumber": ["Pipe installation", "Fixture repairs", "Water heater maintenance", "Blueprint reading"],
        "electrician": ["Wiring installation", "Circuit troubleshooting", "Electrical safety compliance", "Panel upgrades"],
        "nail technician": ["Manicures & Pedicures", "Nail art & design", "Acrylic & gel application", "Sanitation & hygiene"],
        "barber": ["Haircutting techniques", "Shaving & beard styling", "Customer service", "Scalp treatments"],
        "personal trainer": ["Fitness assessments", "Strength training", "Nutritional guidance", "Injury prevention"],
        "house cleaner": ["Deep cleaning", "Organizing & decluttering", "Eco-friendly products", "Carpet stain removal"],
        "chiropractor": ["Spinal adjustments", "Posture correction", "Pain management", "Sports therapy"],
        "resume reviewer": ["ATS optimization", "Grammar & clarity", "Career coaching", "Industry-specific formatting"],
        "tutor": ["Lesson planning", "Student engagement", "Subject mastery", "Test preparation"],
        "graphic designer": ["Adobe Photoshop", "Brand identity", "Typography", "User experience (UX)"],
        "software developer": ["Python & JavaScript", "Backend development", "API integrations", "Database design"]
    }
    return random.sample(skill_map.get(job, ["Communication", "Time Management", "Problem-Solving", "Customer Service"]), 3)

def random_experience(job):
    """Generate unique experience entries for a job."""
    experience_map = {
        "plumber": [
            "Installed plumbing systems for residential homes, reducing maintenance issues by 40%.",
            "Diagnosed and repaired water leakage problems in large-scale commercial buildings.",
            "Led a team of apprentices, providing hands-on training in pipefitting and drainage systems."
        ],
        "electrician": [
            "Designed and implemented safe electrical wiring systems for modern smart homes.",
            "Fixed high-voltage issues in commercial buildings, ensuring compliance with safety codes.",
            "Installed and upgraded circuit breaker panels, improving energy efficiency."
        ],
        "nail technician": [
            "Provided high-end nail art services, increasing client retention by 30%.",
            "Trained junior technicians on the latest acrylic and gel application techniques.",
            "Maintained a 5-star rating through excellent customer service and creative designs."
        ],
        "software developer": [
            "Developed a RESTful API that reduced data processing time by 50%.",
            "Implemented machine learning algorithms to improve fraud detection accuracy.",
            "Collaborated with designers to create a user-friendly interface for a mobile application."
        ],
        "graphic designer": [
            "Designed branding materials for 20+ companies, enhancing their visual identity.",
            "Developed a social media ad campaign that boosted engagement by 40%.",
            "Created UI/UX wireframes for a SaaS product, improving user experience."
        ],
        "tutor": [
            "Improved student test scores by 25% through personalized tutoring sessions.",
            "Designed engaging lesson plans tailored to different learning styles.",
            "Tutored over 50 students in math, helping them achieve their academic goals."
        ],
    }
    return random.choice(experience_map.get(job, ["Provided excellent service and ensured client satisfaction."]))

def generate_random_text(job):
    """Generate a diverse and creative resume text for different jobs."""
    name = random_person_name()
    phone = random_phone()
    email = random_email(name)
    years_exp = random.randint(1, 20)
    skills = ", ".join(random_skills(job))
    experience = random_experience(job)

    base_paragraph = (
        f"Name: {name}\n"
        f"Phone: {phone}\n"
        f"Email: {email}\n\n"
        f"Objective: Passionate and skilled {job} with {years_exp} years of experience, "
        f"seeking to leverage expertise in a dynamic work environment.\n\n"
        f"Skills: {skills}\n\n"
        f"Experience:\n - {experience}\n"
    )
    return base_paragraph

def create_pdf(job, resume_id, output_dir="dummy_resumes"):
    """
    Create a PDF file containing diverse and realistic resume text for different jobs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sanitized_job = job.replace("/", "_").replace(" ", "_")
    job_dir = os.path.join(output_dir, sanitized_job)

    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    resume_text = generate_random_text(job)

    for line in resume_text.split("\n"):
        pdf.cell(200, 10, txt=line, ln=1)

    filename = f"{sanitized_job}_resume_{resume_id}.pdf"
    filepath = os.path.join(job_dir, filename)
    pdf.output(filepath)
    return filepath

def generate_dummy_pdfs():
    """Generate 2-3 dummy PDFs for each job role and return the file paths."""
    jobs = [
        "plumber", "electrician", "nail technician", "barber",
        "personal trainer", "house cleaner", "chiropractor",
        "resume reviewer", "tutor", "graphic designer", "software developer"
    ]
    pdf_files = []
    for job in jobs:
        count = random.randint(2, 3)
        for i in range(count):
            path = create_pdf(job, i)
            pdf_files.append(path)
    return pdf_files


########################################################################
# STEP B: RAG Pipeline Class
########################################################################

class RAGPipeline:
    def __init__(self, mongo_uri, db_name, collection_name, index_name="default"):
        """
        Initialize the pipeline with:
          - Connection details for MongoDB Atlas
          - Database and collection names
          - Index name for vector search
        """
        # -----------------------------------------------------------------
        # Use MongoDB Atlas' recommended approach to create a new client
        # with Server API version 1, plus a quick 'ping' test.
        # -----------------------------------------------------------------
        self.mongo_uri = mongo_uri
        self.client = MongoClient(mongo_uri, server_api=ServerApi('1'))
        
        # Quick ping test (optional, but recommended)
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print("Error pinging MongoDB:", e)

        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        self.index_name = index_name
        
        # Use a small model to save resources:
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim = 384  # all-MiniLM-L6-v2 has dimension 384

    ########################################################
    # 1) PDF Ingestion & Chunking
    ########################################################
    def read_and_chunk_pdf(self, pdf_path, words_per_chunk=150):
        """
        Reads a PDF and splits its text into chunks of ~words_per_chunk words.
        Returns a list of text chunks.
        """
        doc = fitz.open(pdf_path)
        text_chunks = []
        
        for page in doc:
            page_text = page.get_text()
            # Clean up whitespace
            page_text = re.sub(r"\s+", " ", page_text).strip()
            
            # Split words
            words = page_text.split(" ")
            # Group them in chunks
            for i in range(0, len(words), words_per_chunk):
                chunk_words = words[i : i + words_per_chunk]
                chunk_text = " ".join(chunk_words)
                # skip empty or too-short chunks
                if len(chunk_text) < 30:
                    continue
                text_chunks.append(chunk_text)
        
        doc.close()
        return text_chunks

    ########################################################
    # 2) Embed & Store in MongoDB
    ########################################################
    def embed_and_store(self, pdf_files):
        """
        For each PDF, read & chunk -> embed each chunk -> insert into MongoDB.
        """
        # Clear the collection to avoid duplicate key errors from previous runs
        self.collection.delete_many({})

        all_docs = []
        
        for pdf_path in pdf_files:
            chunks = self.read_and_chunk_pdf(pdf_path)
            
            for chunk_text in chunks:
                emb = self.embedding_model.encode(chunk_text)
                emb_list = emb.tolist()  # convert np array -> list for Mongo
                
                # Generate a unique _id using the PDF file name and a UUID
                doc = {
                    "_id": f"{os.path.basename(pdf_path)}_chunk_{uuid.uuid4()}",
                    "pdf_file": os.path.basename(pdf_path),
                    "chunk_text": chunk_text,
                    "embedding": emb_list
                }
                all_docs.append(doc)

        if all_docs:
            # Bulk insert
            self.collection.insert_many(all_docs)
            print(f"Inserted {len(all_docs)} chunk-documents into MongoDB {self.db.name}.{self.collection.name}.")

    ########################################################
    # 3) Vector Search in MongoDB Atlas
    ########################################################
    def search(self, query, top_k=3):
        query_emb = self.embedding_model.encode(query).tolist()

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    # Use "queryVector" here
                    "queryVector": query_emb,
                    "path": "embedding",
                    "limit": top_k,
                    "numCandidates": 50
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "pdf_file": 1,
                    "chunk_text": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(self.collection.aggregate(pipeline))
        return results



    ########################################################
    # 4) Optional: ChatGPT Summarization
    ########################################################
    
    import openai  # Import OpenAI module

    

    def ask_chatgpt(self, query, retrieved_docs, openai_api_key):
        # Fix potential encoding issues in API key
        openai_api_key = openai_api_key.encode("utf-8").decode("utf-8")
        client = openai.Client(api_key=openai_api_key)  # Correct way to initialize the client

        if not retrieved_docs:
            return "No relevant results found in the database."

        # Build context from retrieved documents
        context_text = "\n".join([
            f"Candidate Name: {doc['chunk_text'].split('Phone:')[0].strip()}\n"
            f"Contact Info: {doc['chunk_text'].split('Phone:')[1].strip()}\n"
            f"Skills & Experience: {' '.join(doc['chunk_text'].split('Skills:')[1:]).strip()}"
            for doc in retrieved_docs
        ])

        # System prompt to guide ChatGPT
        system_prompt = (
            "You are a helpful assistant that matches clients with the best professional for their needs. "
            "Use the provided resumes to recommend the most suitable service provider. "
            "Provide a structured answer including the candidate's name, contact details, and why they are a good fit."
        )

        # User prompt with structured query
        user_prompt = (
            f"A user is looking for a professional with this request:\n\n"
            f"QUERY: {query}\n\n"
            f"Here are some candidate profiles:\n"
            f"{context_text}\n\n"
            f"Based on these profiles, recommend the best match with reasoning."
        )

        # Correct OpenAI API call for v1.0+
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the mini model for cost efficiency
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )

        return response.choices[0].message.content





########################################################################
# STEP C: MAIN DEMO USAGE
########################################################################

if __name__ == "__main__":
    # 1) Generate dummy PDF resumes for all jobs
    # dummy_pdf_paths = generate_dummy_pdfs()
    # print(f"Generated {len(dummy_pdf_paths)} dummy PDF resumes in the 'dummy_resumes' folder.")

   # 2) Create a RAG pipeline instance
    #    This is the MongoDB URI from Atlas. Note we use server_api=ServerApi('1') above.
    #    Ensure your <PASSWORD> is correct and IP access is open in Atlas.
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = "testdb"
    COLLECTION_NAME = "chunks"
    INDEX_NAME = "default"  # or the name you used in Atlas for your vector index

    pipeline = RAGPipeline(
        mongo_uri=MONGO_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        index_name=INDEX_NAME
    )

    # 3) Embed and store the new PDFs
    # pipeline.embed_and_store(dummy_pdf_paths)

    # 4) Test searching
    test_query = "I need a personal trainer to help me stay fit."
    top_k = 3
    results = pipeline.search(test_query, top_k=top_k)
    print(f"Search results for query: '{test_query}'\n")
    for r in results:
        print(f"Score: {r['score']:.3f}  |  PDF: {r['pdf_file']}")
        print("Chunk excerpt:", r["chunk_text"][:200], "...\n")

    # (Optional) 5) Summarize with ChatGPT
    # Replace with your real OpenAI API key if you want to test
    openai_api_key = os.getenv("OPENAI_API_KEY")
    final_answer = pipeline.ask_chatgpt(test_query, results, openai_api_key)
    print("ChatGPT Answer:\n", final_answer)

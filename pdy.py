import fitz  # PyMuPDF
import re
from tqdm import tqdm

def read_pdf(pdf_path: str):
    """Reads a PDF and returns a list of page texts."""
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        text = page.get_text()
        pages_text.append(text)
    doc.close()
    return pages_text

def split_into_chunks(text: str, chunk_size=200):
    """
    Split the text into chunks of about `chunk_size` words.
    A naive approach: just split on whitespace and group every `chunk_size` words.
    """
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

# Example usage:
pdf_path = "example.pdf"  # Replace with your local PDF
all_pages = read_pdf(pdf_path)
all_chunks = []

for page_text in all_pages:
    # Optionally do some cleanup (remove weird newlines, etc.)
    page_text = re.sub(r"\s+", " ", page_text).strip()
    
    # Now split the page text
    for chunk in split_into_chunks(page_text, chunk_size=200):  # 200 words per chunk
        if len(chunk) > 50:  # ignore trivial short chunks
            all_chunks.append(chunk)

print(f"Number of chunks: {len(all_chunks)}")
# If you don't have a PDF, just do:
# all_chunks = ["Some text paragraph 1...", "Some text paragraph 2..."]

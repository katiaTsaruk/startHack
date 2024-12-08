import PyPDF2
from transformers import pipeline, TFAutoModelForTokenClassification, AutoTokenizer
from transformers import TokenClassificationPipeline
from plyer import notification  # Import the plyer library for notifications
import subprocess


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDF2."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text


def summarize_text(text, max_length=1024, min_length=50):
    """Summarize text using a pre-trained TensorFlow model (BART)."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn",
                          framework="tf")

    # Ensure the text does not exceed model's token limit
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


def chunk_text(text, chunk_size=1024):
    """Chunks the text into smaller parts to avoid model token limit issues."""
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def extract_key_phrases(text):
    """Extract key phrases using Hugging Face's TensorFlow-compatible KeyPhrase Extraction pipeline."""
    model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForTokenClassification.from_pretrained(model_name, from_pt=True)
    extractor = TokenClassificationPipeline(model=model, tokenizer=tokenizer, aggregation_strategy="simple",
                                            framework="tf")

    return extractor(text)



def show_notification(summary_text):
    """Show a desktop notification with the summary text using notify-send."""
    try:
        # Limit the message length for better readability
        truncated_message = summary_text[:200]  # Show the first 200 characters
        subprocess.run(['notify-send', 'Company Report Summary', truncated_message])
    except Exception as e:
        print(f"Error displaying notification: {e}")



# Example Usage
pdf_path = "../Q3_2023_Earnings_Presentation.pdf"  # Path to the PDF file
full_text = extract_text_from_pdf(pdf_path)

# If the text is too long, split it into chunks and summarize each chunk
chunks = chunk_text(full_text, chunk_size=1024)

summaries = []
for chunk in chunks:
    summary = summarize_text(chunk, max_length=1024, min_length=50)
    summaries.append(summary)

# Combine summaries from all chunks
final_summary = " ".join(summaries)

# Print the final summary
print("Final Summary:")
print(final_summary)

# Show the summary as a desktop notification
show_notification(final_summary)

# Example Usage
#pdf_path = "Q3_2023_Earnings_Presentation.pdf"  # Path to the PDF file

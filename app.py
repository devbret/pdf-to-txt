import os
import PyPDF2
from datetime import datetime

LOG_FILE = "pdf_extraction.log"
OUTPUT_FILE = "combined_output.txt"

def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(entry)

def ensure_log_file_exists():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as log_file:
            log_file.write("PDF Extraction Log\n==================\n\n")

def ensure_output_file_exists():
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("")
        log(f"Output file prepared: {OUTPUT_FILE}")
    except Exception as e:
        error_msg = f"Failed to prepare output file: {e}"
        print(error_msg)
        log(error_msg)
        raise

def extract_text_from_pdfs(directory="./input"):
    ensure_log_file_exists()
    log("Extraction started.")
    ensure_output_file_exists()
    
    output_text = []
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]

    if not pdf_files:
        msg = "No PDF files found in the directory."
        print(msg)
        log(msg)
        return

    print(f"Found {len(pdf_files)} PDF file(s). Starting extraction...\n")
    log(f"Found {len(pdf_files)} PDF file(s).")

    for filename in pdf_files:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                print(f"Extracting from: {filename} ({num_pages} pages)")
                log(f"Extracting from: {filename} ({num_pages} pages)")
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        output_text.append(text)
        except Exception as e:
            error_msg = f"[ERROR] Could not read {filename}: {e}"
            print(error_msg)
            log(error_msg)

    combined_text = "\n\n".join(output_text)
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(combined_text)
        print(f"\nExtraction complete. Saved to: {OUTPUT_FILE}")
        log(f"Extraction complete. Saved to: {OUTPUT_FILE}")
    except Exception as e:
        error_msg = f"Failed to write output file: {e}"
        print(error_msg)
        log(error_msg)

    log("Extraction finished.\n")

if __name__ == "__main__":
    extract_text_from_pdfs()


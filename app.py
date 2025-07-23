import os
import PyPDF2

def extract_text_from_pdfs(directory="./input"):
    output_text = []
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in the directory.")
        return

    print(f"Found {len(pdf_files)} PDF file(s). Starting extraction...\n")

    for filename in pdf_files:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                print(f"Extracting from: {filename} ({len(reader.pages)} pages)")
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        output_text.append(text)
        except Exception as e:
            print(f"[ERROR] Could not read {filename}: {e}")

    combined_text = "\n\n".join(output_text)
    output_file = os.path.join(".", "combined_output.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_text)

    print(f"\nExtraction complete. Saved to: {output_file}")

if __name__ == "__main__":
    extract_text_from_pdfs()

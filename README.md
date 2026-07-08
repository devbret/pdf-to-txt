# PDF-To-TXT

Flexible bulk PDF text extraction tool combining multiple processing engines, parallel processing, consolidated outputs and detailed logging into one workflow.

## Application Overview

PDF-To-TXT is a powerful and flexible Python tool for extracting text from PDF files in bulk. It supports both the `PyPDF` and `pdfminer.six` engines for text extraction, with automatic fallback to OCR using `Tesseract` when standard extraction fails.

The program scans a specified input directory for `PDF` files and can process documents in parallel across multiple CPU cores. Each extraction run produces a consolidated text output file (`combined_output.txt`) and can also generate per-file text outputs as well as a detailed `JSONL` index containing metadata.

The script provides extensive logging and progress tracking, recording every action to both the console and a log file (`pdf_extraction.log`). Users can specify page ranges, choose between extraction engines, enable/disable OCR and fine-tune performance with worker counts.

Whether used for lightweight text gathering or large-scale document processing, PDF-To-TXT delivers a reliable and transparent workflow with detailed logs, clear error reporting and flexible configuration options.

## Basic Setup Instructions

Below are instructions for installing and running this application on a Linux machine.

### Programs Needed

- [Git](https://git-scm.com/downloads)

- [Python](https://www.python.org/downloads/)

### Steps

1. Install the above programs

2. Open a terminal

3. Clone this repository: `git clone git@github.com:devbret/pdf-to-txt.git`

4. Navigate to the repo's directory: `cd pdf-to-txt`

5. Create a virtual environment: `python3 -m venv venv`

6. Activate your virtual environment: `source venv/bin/activate`

7. Install the needed dependencies: `pip install -r requirements.txt`

8. Place your `PDF` files in the `input` directory of this repo

9. Run the application: `python3 app.py`

10. When finished, exit the virtual environment: `deactivate`

## Other Considerations

This project repo is intended to demonstrate an ability to do the following:

- Extract text from any number of PDF files in bulk and consolidate the results into a single `.txt` file

- Choose between the `pypdf` and `pdfminer.six` extraction engines

- Process PDFs in parallel across multiple CPU cores, with configurable worker counts and page-range selection

- Record runs in a detailed log file and an optional `JSONL` metadata index and finish with a summary

If you have any questions or would like to collaborate, please reach out either on GitHub or via [my website](https://bretbernhoft.com/).

# PDF-To-Txt

This script extracts and combines text from all PDF files in a specified directory using the PyPDF2 library. It reads each page of each PDF, collects the extracted text, and saves the combined result to a file named `combined_output.txt`. Before extraction begins, the script ensures the output file exists (or is created) and all actions are logged with timestamps in `pdf_extraction.log`. It includes error handling for unreadable PDFs and provides console output to indicate progress throughout execution.

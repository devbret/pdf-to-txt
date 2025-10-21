# PDF-To-Txt

PDF-To-Txt is a powerful and flexible Python tool for extracting text from PDF files in bulk. It supports both the PyPDF and pdfminer.six engines for text extraction, with automatic fallback to OCR using Tesseract when standard extraction fails.

The program scans a specified input directory for PDFs, and can process files in parallel across multiple CPU cores. Each extraction produces a consolidated text output file (`combined_output.txt`) and can also generate per-file text outputs and a detailed JSONL index containing metadata.

The script provides extensive logging and progress tracking, recording every action to both the console and a log file (`pdf_extraction.log`). Users can specify page ranges, choose between extraction engines, enable or disable OCR and fine-tune performance with custom worker counts.

Whether used for lightweight text gathering or large-scale document processing, PDF-To-Txt delivers a reliable and transparent workflow with detailed logs, clear error reporting and flexible configuration options.

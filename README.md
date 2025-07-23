# PDF-To-Txt

Extracts and combines text from all PDF files in a specified directory. It uses the PyPDF2 library to read each PDF, extract text from each page, and appends the resulting text into a list. After processing all PDFs, it joins the extracted text into a single string with double line breaks between entries and saves it to a file named combined_output.txt in the current working directory. The script includes error handling for unreadable files and prints status updates during execution.

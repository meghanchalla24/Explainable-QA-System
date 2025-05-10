import os
import fitz  
import pytesseract
from PIL import Image
import io
import json
# Setting tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\User\AppData\Local\Programs\Tesseract-OCR\tesseract.exe" 

# Base URL for PDFs and images
BASE_URL = "https://www.bits-pilani.ac.in/wp-content/uploads/"

# Custom links for specific files
CUSTOM_LINKS = {
    "GCIR SOP_Hyd_11oct.pdf": "https://drive.google.com/file/d/14QQf8xW5rkj3ifiIGTbbBK0noCaNpCE0/view",
    "Overhead_Distributon_Policy_01_APRIL 2024.pdf":"https://drive.google.com/file/d/1LXA3RAOOclkfNn1cpDZGdNeoD1eZ-InW/view",
    "PhD-Guideline-Brochure_2019.pdf":"https://universe.bits-pilani.ac.in/uploads/PhD%20Guideline%20Brochure_2019.pdf"
}


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF (fitz). Uses OCR if necessary."""
    try:
        doc = fitz.open(pdf_path)
        extracted_pages = []
        is_scanned = False

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()

            if not text:  # If no text, apply OCR
                is_scanned = True
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, lang="eng").strip()

            extracted_pages.append({"page": page_num + 1, "text": text})

        return extracted_pages, is_scanned
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return [], False


def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    try:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img, lang="eng").strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""


def process_files(pdf_folder=r"D:\Information_retrieval_project\data\raw\corpus_pdfs", output_file="extracted_data_with_links.json", links_output_file="pdf_links.txt"): #optional - change your default folder location accordingly
    """Process all PDF and image files in a folder, extract text, and save to JSON and text files."""
    if not os.path.exists(pdf_folder):
        print(f"Error: The folder '{pdf_folder}' does not exist.")
        return

    pdf_data = {}
    all_links = []

    for file_name in os.listdir(pdf_folder):
        file_path = os.path.join(pdf_folder, file_name)

        if file_name.lower().endswith(".pdf"):
            extracted_pages, is_scanned = extract_text_from_pdf(file_path)
            if extracted_pages:
                source_link = CUSTOM_LINKS.get(file_name, f"{BASE_URL}{file_name.replace(' ', '-')}")
                pdf_data[file_name] = {
                    "local_path": file_path,
                    "source_link": source_link,
                    "is_scanned": is_scanned,
                    "extracted_text": extracted_pages
                }
                all_links.append(f"Local Path: {file_path}\nSource Link: {source_link}\n")

        elif file_name.lower().endswith(".jpg"):  # Handling image files
            extracted_text = extract_text_from_image(file_path)
            if extracted_text:
                source_link = f"{BASE_URL}{file_name.replace(' ', '-')}"
                pdf_data[file_name] = {
                    "local_path": file_path,
                    "source_link": source_link,
                    "is_scanned": True,
                    "extracted_text": [{"page": 1, "text": extracted_text}]
                }
                all_links.append(f"Local Path: {file_path}\nSource Link: {source_link}\n")

    # Save extracted text to JSON
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pdf_data, f, indent=4, ensure_ascii=False)
        print(f"Text extracted and saved to {output_file}")
    except Exception as e:
        print(f"Error saving extracted text: {e}")

    # Save links to a text file
    try:
        with open(links_output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(all_links))
        print(f"List of links saved to {links_output_file}")
    except Exception as e:
        print(f"Error saving links: {e}")

    return output_file


if __name__ == "__main__":
    print("This module should be imported, not run directly.")

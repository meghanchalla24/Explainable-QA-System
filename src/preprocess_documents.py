import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import difflib


def data_clean(input_file=r"data\processed\final_corrected_extracted_data.json"): # change your default accordingly
    # Load the extracted JSON data
    output_file = "cleaned_extracted_data.json"

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            pdf_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        exit()

    # Initialize NLP tools
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    exceptions = {"bits"} # we will add more words if needed as per the results


    def clean_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"_+", " ", text)  # Replace multiple underscores with a space
        text = re.sub(r"(?<!\d)\.(?!\d)|[^\w\s.]", "", text)  # Remove punctuation - make sure 1.5 is not converted to 15
        text = " ".join(text.split())  # Remove extra spaces
        tokens = word_tokenize(text)  # Tokenization
        tokens = [word for word in tokens if word not in stop_words]  # Stopword removal
        tokens = [word if word in exceptions else lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization with exceptions
        return " ".join(tokens)


    # Process each PDF entry and clean the text while maintaining structure
    for pdf_file, data in pdf_data.items():
        cleaned_pages = []
        
        for page in data["extracted_text"]:
            if isinstance(page, dict) and "text" in page and "page" in page:
                cleaned_page = {
                    "page": page["page"],
                    "text": clean_text(page["text"]) if isinstance(page["text"], str) else page["text"]
                }
                cleaned_pages.append(cleaned_page)
            else:
                cleaned_pages.append(page)  # Preserve original structure if unexpected format

        data["extracted_text"] = cleaned_pages

    # Save the cleaned data to a new JSON file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pdf_data, f, indent=4, ensure_ascii=False)
        print(f"Cleaned data successfully saved to '{output_file}'!")
    except Exception as e:
        print(f"Error saving cleaned JSON file: {e}")

    return output_file



def spell_correction(input_file=r"data\raw\raw_json\extracted_data_with_links.json"):
    with open(input_file, "r", encoding="utf-8") as file:
        extracted_details = json.load(file)
        
    exceptions = {"bits"} # we will add more words if needed as per the results

    output_file = "final_corrected_extracted_data.json"

    # Example dictionary (you can expand this list)
    dictionary = ["scholars", "expenses", "attending", "academic", "conferences", "contingency", "guidelines", "items", "procured", 
                "grant", "institute", "fellowship", "research", "books", "journals", "stationary", "calculator", "laser","pointer", 
                "laptop" ,"mouse", "cooling", "pad", "printer", "cartridges", "supplies", "computer", "consumables", "external", "storage", 
                "device", "battery", "anti-virus", "software", "sample", "participation", "professional", "development", "programs", 
                "chemicals", "glassware", "photocopying", "typing", "binding", "charges", "spare", "parts", "replacement", "minor", 
                "repair", "laptops", "recording", "spectra", "experimental", "facility", "utilization", "consumable", "materials", 
                "required", "theoretical", "studies", "data", "cards", "purchases", "April", "maximum", "five", "years", "until", "VIva-Voce","like","equivalent",
                "programme"]

    def spell_checker(text):

        words = text.split()
        corrected_words = []

        for word in words:
            if word in dictionary or word in exceptions:
                corrected_words.append(word)
            else:
                # Find the closest match using difflib
                closest_match = difflib.get_close_matches(word, dictionary, n=1)
                corrected_words.append(closest_match[0] if closest_match else word)
        
        return " ".join(corrected_words)


    # Function to process the extracted details and apply spell checker for scanned documents
    def process_extracted_details(extracted_details):
        for file_name, document in extracted_details.items():
            if file_name.lower().endswith(".jpg"):  # Apply spell checking only if the file is a .jpg
                if document.get("is_scanned"):  # Apply spell checking only if scanned
                    for page in document.get("extracted_text", []):
                        # Apply spell checker to the extracted text
                        page['text'] = spell_checker(page['text'])
        
        return extracted_details

    # Process the extracted details
    processed_details = process_extracted_details(extracted_details)

    # Now, `processed_details` contains the text with corrected spellings for scanned .jpg files.

    with open(output_file, "w") as file:
        json.dump(processed_details, file, indent=4)
        
    return output_file

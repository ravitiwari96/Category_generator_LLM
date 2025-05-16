import os
import json
import uuid
import re
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import csv
import io

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")


genai.configure(api_key=API_KEY)

def extract_json_from_text(text):
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            raise ValueError("Could not extract valid JSON from the response.")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON: {e}")
        raise ValueError("Invalid JSON format in extracted text.")

def get_extracted_keywords(natural_input):
    
    try:
        extract_prompt = f"""
        From the following sentence, extract two lists:
        1. Positive prompts (categories to include)
        2. Negative prompts (categories to avoid)

        Return JSON format like this:
        {{"positive": [...], "negative": [...]}}

        Sentence: "{natural_input}"
        """
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        extraction_response = model.generate_content(extract_prompt)
        extracted_text = extraction_response.text.strip()

        if extracted_text.startswith("```"):
            extracted_text = extracted_text.split("```")[1].strip()

        extracted_json = extract_json_from_text(extracted_text)
        print({"positive": extracted_json["positive"], "negative": extracted_json["negative"]})
        return {"positive": extracted_json["positive"], "negative": extracted_json["negative"]}
    
    except Exception as e:
        logging.error(f"Error extracting keywords: {e}")
        raise

def build_category_prompt(positive, negative=None):
  
    try:
        if not positive or not isinstance(positive, list):
            raise ValueError("Positive keywords must be a non-empty list.")

        positive_str = ", ".join(positive)
        prompt_lines = [
            "You are an expert in e-commerce category generation.",
            "",
            "Generate a detailed product category tree in CSV format with the headers:",
            "Category ID,Parent ID,Category Name.",
            "",
            f"Use the following keywords to guide the content:",
            f"Positive keywords: {positive_str}"
        ]

        if negative and isinstance(negative, list) and negative:
            negative_str = ", ".join(negative)
            prompt_lines.append(f"Exclude anything related to the following negative keywords: {negative_str}")

        prompt_lines += [
            "",
            "Ensure the structure is fully detailed and practical for a real online store.",
            "The structure should form a hierarchy using Parent ID.",
            "Root categories should have Parent ID = 0.",
            "",
            "Output ONLY the CSV content. Do not include any commentary, markdown formatting, or code block."
        ]

        return "\n".join(prompt_lines)

    except Exception as e:
        logging.error(f"Error building category prompt: {e}")
        raise


def clean_csv(csv_content, negative_keywords_str):
    

    reader = csv.reader(io.StringIO(csv_content))
    headers = next(reader)

    if negative_keywords_str:
        negative_keywords = [word.strip().lower() for word in negative_keywords_str.split(",") if word.strip()]
    else:
        negative_keywords = []

    output = [headers]
    for row in reader:
        category_name = row[2].lower()
        if not any(neg in category_name for neg in negative_keywords):
            output.append(row)

    output_io = io.StringIO()
    writer = csv.writer(output_io)
    writer.writerows(output)
    return output_io.getvalue()


def save_csv_file(content):
   
    try:
        filename = f"category_tree_{uuid.uuid4()}.csv"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"CSV file saved as {filename}")
        return filename
    except IOError as e:
        logging.error(f"Error saving CSV file: {e}")
        raise

# Main execution flow
def main():
    try:
       
        natural_input = input("🗣️ Describe your shop's product focus (e.g. what to include or avoid):\n> ")
        
        if not natural_input.strip():
            raise ValueError("Please enter your requirements.")
        
        
        keywords = get_extracted_keywords(natural_input)
        positive, negative = keywords["positive"], keywords["negative"]

        
        category_prompt = build_category_prompt(positive, negative)
        print(f"📝 Generating category tree with the following prompt:\n{category_prompt}")
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(category_prompt)
        csv_output = response.text.strip()

        if csv_output.startswith("```"):
            csv_output = csv_output.split("```")[1].strip()

        
        cleaned_csv = clean_csv(csv_output, ",".join(negative))

        
        saved_filename = save_csv_file(cleaned_csv)
        print(f"Category tree saved to '{saved_filename}'.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"\n Error: {e}")

if __name__ == "__main__":
    main()

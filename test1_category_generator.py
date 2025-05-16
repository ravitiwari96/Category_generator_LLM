import unittest
from unittest.mock import patch, MagicMock, mock_open
import builtins
import json
import re
import io
import csv

# Import the functions from your module here.
# Assuming your module file is named `category_generator.py`
from category_generator import (
    extract_json_from_text,
    get_extracted_keywords,
    build_category_prompt,
    clean_csv,
    save_csv_file,
    main
)

# For demonstration, I'll assume functions are defined in this script
# You should replace the above import with your actual module import


class TestCategoryGenerator(unittest.TestCase):

    def test_extract_json_from_text_valid(self):
        text = 'Here is some text {"positive": ["cat1", "cat2"], "negative": ["cat3"]} and more text.'
        expected = {"positive": ["cat1", "cat2"], "negative": ["cat3"]}
        result = extract_json_from_text(text)
        self.assertEqual(result, expected)

    def test_extract_json_from_text_invalid_json(self):
        text = 'Here is text {invalid json: true,} text'
        with self.assertRaises(ValueError) as context:
            extract_json_from_text(text)
        self.assertIn("Invalid JSON format", str(context.exception))

    def test_extract_json_from_text_no_json(self):
        text = 'No json here!'
        with self.assertRaises(ValueError) as context:
            extract_json_from_text(text)
        self.assertIn("Could not extract valid JSON", str(context.exception))

    @patch('google.generativeai.GenerativeModel')
    def test_get_extracted_keywords_success(self, mock_model_cls):
        # Mock the model and its return value
        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = '```{"positive": ["electronics"], "negative": ["clothing"]}```'
        mock_model_cls.return_value = mock_model

        natural_input = "I want electronics but no clothing"
        result = get_extracted_keywords(natural_input)
        self.assertEqual(result, {"positive": ["electronics"], "negative": ["clothing"]})

    @patch('google.generativeai.GenerativeModel')
    def test_get_extracted_keywords_error(self, mock_model_cls):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API failure")
        mock_model_cls.return_value = mock_model

        with self.assertRaises(Exception) as context:
            get_extracted_keywords("test input")
        self.assertIn("API failure", str(context.exception))

    def test_build_category_prompt_positive_only(self):
        positive = ["electronics", "gadgets"]
        prompt = build_category_prompt(positive)
        self.assertIn("Positive keywords: electronics, gadgets", prompt)
        self.assertNotIn("Exclude anything related", prompt)

    def test_build_category_prompt_with_negative(self):
        positive = ["electronics"]
        negative = ["clothing"]
        prompt = build_category_prompt(positive, negative)
        self.assertIn("Exclude anything related to the following negative keywords: clothing", prompt)

    def test_build_category_prompt_invalid_positive(self):
        with self.assertRaises(ValueError):
            build_category_prompt(None)

        with self.assertRaises(ValueError):
            build_category_prompt([])

        with self.assertRaises(ValueError):
            build_category_prompt("not a list")

    def test_clean_csv_basic(self):
        csv_content = "Category ID,Parent ID,Category Name\n1,0,Electronics\n2,1,Clothing\n3,1,Toys\n"
        negative_keywords = "clothing"
        cleaned = clean_csv(csv_content, negative_keywords)
        # Should exclude Clothing row
        self.assertIn("Electronics", cleaned)
        self.assertIn("Toys", cleaned)
        self.assertNotIn("Clothing", cleaned)

    def test_clean_csv_empty_negative(self):
        csv_content = "Category ID,Parent ID,Category Name\n1,0,Electronics\n"
        cleaned = clean_csv(csv_content, "")
        self.assertIn("Electronics", cleaned)

    @patch("builtins.open", new_callable=mock_open)
    @patch("uuid.uuid4")
    def test_save_csv_file_success(self, mock_uuid, mock_file):
        mock_uuid.return_value = "1234"
        content = "Category ID,Parent ID,Category Name\n1,0,Electronics\n"
        filename = save_csv_file(content)
        self.assertTrue(filename.startswith("category_tree_1234.csv"))
        mock_file.assert_called_once_with(filename, "w", encoding="utf-8")
        handle = mock_file()
        handle.write.assert_called_once_with(content)

    @patch("builtins.open", side_effect=IOError("Disk error"))
    def test_save_csv_file_io_error(self, mock_file):
        with self.assertRaises(IOError):
            save_csv_file("data")

    @patch("builtins.input", return_value="Include electronics and gadgets, avoid clothing")
    @patch('google.generativeai.GenerativeModel')
    @patch("builtins.open", new_callable=mock_open)
    @patch("uuid.uuid4", return_value="1234")
    def test_main_success(self, mock_uuid, mock_file, mock_genai_model, mock_input):
        # Mock the Gemini API for keyword extraction
        mock_model_instance = MagicMock()
        # First call: extract keywords
        mock_model_instance.generate_content.side_effect = [
            MagicMock(text='```{"positive": ["electronics", "gadgets"], "negative": ["clothing"]}```'),
            MagicMock(text='''Category ID,Parent ID,Category Name
1,0,Electronics
2,1,Toys
3,1,Clothing
''')
        ]
        mock_genai_model.return_value = mock_model_instance

        # Run main (should not raise)
        with patch('builtins.print') as mock_print:
            main()

        # Check print calls include saved filename
        printed = [args[0] for args, _ in mock_print.call_args_list]
        found = any("Category tree saved to 'category_tree_1234.csv'." in p for p in printed)
        self.assertTrue(found)

    @patch("builtins.input", return_value="")
    def test_main_empty_input(self, mock_input):
        with patch('builtins.print') as mock_print:
            main()
            printed = [args[0] for args, _ in mock_print.call_args_list]
            found = any("Please enter your requirements." in p for p in printed)
            self.assertTrue(found)
    

if __name__ == "__main__":
    unittest.main()


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from category_generator import extract_json_from_text, build_category_prompt, clean_csv


# ---------------------------
# Test extract_json_from_text
# ---------------------------

def test_extract_json_valid():
    text = '''
    Sure, here is the JSON:
    {
        "positive": ["electronics", "gadgets"],
        "negative": ["clothing", "toys"]
    }
    '''
    result = extract_json_from_text(text)
    assert result["positive"] == ["electronics", "gadgets"]
    assert result["negative"] == ["clothing", "toys"]

def test_extract_json_invalid():
    text = "This is not a JSON."
    with pytest.raises(ValueError):
        extract_json_from_text(text)

# ----------------------------
# Test build_category_prompt
# ----------------------------

def test_build_category_prompt_with_negative():
    prompt = build_category_prompt(["electronics", "gadgets"], ["clothing"])
    assert "electronics" in prompt
    assert "clothing" in prompt
    assert "Exclude anything related" in prompt

def test_build_category_prompt_without_negative():
    prompt = build_category_prompt(["books", "education"], None)
    assert "books" in prompt
    assert "Exclude anything related" not in prompt

def test_extract_json_with_surrounding_text():
    text = """
    Here's some intro text.

    {
        "positive": ["fashion", "shoes"],
        "negative": ["electronics"]
    }

    And some trailing text.
    """
    result = extract_json_from_text(text)
    assert result["positive"] == ["fashion", "shoes"]
    assert result["negative"] == ["electronics"]

def test_extract_json_positive_only():
    text = '''
    {
        "positive": ["home", "furniture"]
    }
    '''
    result = extract_json_from_text(text)
    assert result["positive"] == ["home", "furniture"]
    assert "negative" not in result


def test_build_category_prompt_empty_positive():
    with pytest.raises(ValueError, match="Positive keywords must be a non-empty list."):
        build_category_prompt([], ["food"])


def test_build_category_prompt_empty_both():
    with pytest.raises(ValueError, match="Positive keywords must be a non-empty list."):
        build_category_prompt([], [])


# ----------------------------
# Test clean_csv
# ----------------------------

def test_clean_csv_excludes_negatives():
    csv_data = """Category ID,Parent ID,Category Name
1,0,Electronics
2,1,Toys
3,1,Clothing
4,0,Books"""
    cleaned = clean_csv(csv_data, "Toys,Clothing")
    assert "Toys" not in cleaned
    assert "Clothing" not in cleaned
    assert "Electronics" in cleaned
    assert "Books" in cleaned

def test_clean_csv_all_clean():
    csv_data = """Category ID,Parent ID,Category Name
1,0,Electronics
2,1,Gadgets"""
    cleaned = clean_csv(csv_data, "Clothing")
    lines = cleaned.strip().splitlines()
    # assert cleaned.count("\n") == 2  # header + 2 lines
    assert len(lines) == 3

def test_clean_csv_partial_match_not_removed():
    csv_data = """Category ID,Parent ID,Category Name
1,0,Cloth
2,0,Clothing Accessories
3,0,Books"""
    cleaned = clean_csv(csv_data, "Clothing")
    assert "Cloth" in cleaned  # Should not remove 'Cloth'
    assert "Clothing Accessories" not in cleaned
    assert "Books" in cleaned

def test_clean_csv_case_insensitive():
    csv_data = """Category ID,Parent ID,Category Name
1,0,clothing
2,0,Toys
3,0,Electronics"""
    cleaned = clean_csv(csv_data, "Clothing,TOYS")
    assert "clothing" not in cleaned
    assert "Toys" not in cleaned
    assert "Electronics" in cleaned

def test_clean_csv_no_negatives():
    csv_data = """Category ID,Parent ID,Category Name
1,0,Sports
2,0,Outdoors"""
    cleaned = clean_csv(csv_data, "")
    assert "Sports" in cleaned
    assert "Outdoors" in cleaned

# ----------------------------
# Edge cases
# ----------------------------

def test_clean_csv_empty_input():
    with pytest.raises(Exception):
        clean_csv("", "test")

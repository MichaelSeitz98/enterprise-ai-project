# Generic methods  to apply regex and clean text
import re
import csv
import tempfile
import shutil

def extract_first_match(text, regex_pattern):
    pattern = re.compile(regex_pattern)
    match = pattern.search(text)
    if match:
        return match.group(1)
    else:
        return None
    
def extract_LineMatch(text, regex_pattern):
    pattern = re.compile(regex_pattern, re.MULTILINE)
    match = pattern.search(text)
    if match:
        return match.group()
    else:
        return None
    
def clean_entries(matches, patternsToRemove):
    cleaned_entries = []
    for entry in matches:
        cleaned_text = entry
        for pattern in patternsToRemove:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.MULTILINE)
        cleaned_entries.append(cleaned_text)

    return cleaned_entries

def row_exists_in_csv(filename, row_data):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for existing_row in reader:
            if existing_row == row_data:
                return True
    return False

def add_row_to_csv(filename, row_data):
    if not row_exists_in_csv(filename, row_data):
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row_data)


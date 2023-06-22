# Generic methods  to apply regex and clean text
import re
import csv
import os

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


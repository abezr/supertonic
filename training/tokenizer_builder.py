import json
import os
import urllib.request

def build_extended_indexer():
    # URL to original indexer
    url = "https://huggingface.co/Supertone/supertonic-2/raw/main/onnx/unicode_indexer.json"
    print(f"Downloading original indexer from {url}...")
    
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
    
    # data is a list where index corresponds to unicode integer, value is the ID.
    # Find current max ID to start appending from
    current_ids = [x for x in data if x != -1]
    max_id = max(current_ids) if current_ids else 0
    print(f"Current Max ID: {max_id}")
    
    next_id = max_id + 1
    
    # Extend list if needed to cover Cyrillic
    # Cyrillic block is roughly 0x0400 (1024) to 0x04FF (1279)
    # Plus some extensions for Ukrainian like 0x0490 (Ґ) and 0x0491 (ґ)
    # Let's ensure list is long enough
    
    needed_length = 0x0500 # Cover up to 1280
    if len(data) < needed_length:
        print(f"Extending list from {len(data)} to {needed_length}...")
        data.extend([-1] * (needed_length - len(data)))
        
    cyrillic_ranges = [
        (0x0400, 0x04FF), # Basic Cyrillic
        (0x0500, 0x052F), # Supplementary
        (0x02D0, 0x02D0), # Triangular colon (often used in phonetic) - optional
    ]
    
    # Specific Ukrainian chars check:
    # Є (0404), І (0406), Ї (0407), Ґ (0490)
    # є (0454), і (0456), ї (0457), ґ (0491)
    # unique chars in RU/UK. 
    
    added_count = 0
    characters_added = []
    
    for start, end in cyrillic_ranges:
        # Extend loop to cover extended length if we didn't before
        if len(data) <= end:
             data.extend([-1] * (end - len(data) + 1))
             
        for code in range(start, end + 1):
            if data[code] == -1:
                data[code] = next_id
                next_id += 1
                added_count += 1
                characters_added.append(chr(code))
    
    print(f"Added {added_count} new characters.")
    print(f"New Max ID: {next_id - 1}")
    
    output_path = "unicode_indexer_expanded.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    
    print(f"Saved to {output_path}")
    
    # Check Ukrainian specific
    uk_test_chars = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЮЯабвгґдеєжзиіїйклмнопрстуфхцчшщюя"
    print("\nVerifying Ukrainian characters coverage:")
    missing = []
    for char in uk_test_chars:
        code = ord(char)
        if code >= len(data) or data[code] == -1:
            missing.append(char)
    
    if missing:
        print(f"WARNING: Missing characters: {missing}")
    else:
        print("All Ukrainian characters covered.")

if __name__ == "__main__":
    build_extended_indexer()

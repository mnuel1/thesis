import os
import json

def classify_files(folder_path):
    class_values = {}

    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            # Convert to lowercase for case-insensitive comparison
            lower_filename = filename.lower()
            if "peso" in lower_filename:
                class_values[filename] = 0
            elif "yen" in lower_filename:
                class_values[filename] = 1

    result = {"class_values": class_values}
    return result

# Set your folder path here
folder_path = "./test/train"

# Get the classification
classified_data = classify_files(folder_path)

# Output to JSON
with open("classified_files.json", "w") as json_file:
    json.dump(classified_data, json_file, indent=2)

print("JSON saved as classified_files.json")

import json
import os

input_json = "export/result.json"
output_json = "export/result.json"

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

for img in data["images"]:
    img["file_name"] = os.path.basename(img["file_name"])

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("saved")

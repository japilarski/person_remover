import requests
import os

output_dir = "external_utils"
os.makedirs(output_dir, exist_ok=True)

urls = [
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py"
]

for url in urls:
    filename = url.split('/')[-1]
    response = requests.get(url)

    if response.status_code == 200:
        # Save the file to the utils directory
        with open(os.path.join(output_dir, filename), "wb") as file:
            file.write(response.content)
        print(f"Downloaded {filename} to {output_dir}")
    else:
        print(f"Failed to download {filename}")

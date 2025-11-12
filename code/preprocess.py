import os
import sys
from PIL import Image
from io import BytesIO

from datasets import load_dataset, DatasetDict, Dataset
from datasets import Image as HFDImage
from tqdm import tqdm
import numpy as np


class PreprocessDataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        self.dataset = load_dataset(self.dataset_name, cache_dir=f"../data/raw/{self.dataset_name}")

    def preprocess_sample(self, sample):
        hint = sample.get('hint', None)
        lecture = sample.get('example', None)
        solution = sample.get('solution', None)
        answer = sample.get('answer', None)
        question = sample.get('question', None)
        choices = sample.get('choices', None)
        image = sample.get('image', None)


        if isinstance(image, dict) and 'bytes' in image:
            image_bytes = image['bytes']
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

        if not image:
            image = Image.new("RGB", (224, 224), color=(255, 255, 255)) # Just some filler image

        image = self.resize_and_pad(image)

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        buffer.close()

        prompt = f"<start_of_turn>user\n<start_of_image>\nQuestion: {question}\n"

        for i, choice in enumerate(choices):
            prompt += f"({i}) {choice}\n"

        if hint:
            prompt += f"Hint: {hint}\n"

        prompt += "<end_of_turn>\n"

        target = "<start_of_turn>model\nLet's think through this step by step. I'll first analyze any relevant background. Then I'll continue by doing any necessary reasoning in a solution step. Finally, I'll output the final solution."
        if lecture:
            target += f"Background: {lecture}\n"

        if solution:
            target += f"Solution: {solution}\n"

        if answer:
            target += f"Based on my previous reasoning and the provided answer choices, my answer is {answer}"

        target += "<end_of_turn>"

        return {"prompt": prompt, "target": target, "image": image_bytes}
    
    def resize_and_pad(self, image, size=(224,224), pad_color=(255,255,255)):
        image = image.convert("RGB")
        w, h = image.size
        target_w, target_h = size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        new_image = Image.new("RGB", size, pad_color)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        return new_image
    
    def preprocess_dataset(self):
        dataset = {}
        for split in ['train', 'validation', 'test']:
            processed = []
            for sample in tqdm(self.dataset[split],
                               desc=f"Processing examples for {self.dataset_name}, split {split}",
                               total=len(self.dataset[split])):
                processed.append(self.preprocess_sample(sample))

            dataset[split] = Dataset.from_list(processed)
            dataset[split] = dataset[split].cast_column('image', HFDImage(decode=True))

        dataset = DatasetDict(dataset)

        os.makedirs(f"../data/processed/{self.dataset_name}", exist_ok=True)
        dataset.save_to_disk(f"../data/processed/{self.dataset_name}")

def main():
    DATASET_NAME = "derek-thomas/ScienceQA"
    pd = PreprocessDataset(dataset_name=DATASET_NAME)
    pd.preprocess_dataset()

if __name__ == '__main__':
    main()
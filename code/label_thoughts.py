import json
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from datasets import load_dataset, DatasetDict
import os
import sys
import polars as pl
import re
from tqdm import tqdm
import gc
import argparse

from PIL.Image import Image
from typing import List

CACHE_DIR = "/fs/nexus-scratch/adas1236/.cache/huggingface"

os.environ["HF_HOME"] = CACHE_DIR
# os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

class LabelThoughts:
    def __init__(self, model_name="Qwen/Qwen3-VL-8B-Instruct",
                 taxonomy = None, dataset_name = "multimodal-reasoning-lab/Zebra-CoT",
                 cache_dir = "/fs/nexus-scratch/adas1236/.cache/huggingface"):
        if taxonomy is None:
            taxonomy = [
                "Perception",
                "Spatial Reasoning",
                "Attribute Extraction",
                "Counting",
                "Comparison",
                "Logical Inference",
                "Task Understanding",
                "Answer Construction"
            ]

        self.model_name = model_name
        self.taxonomy = taxonomy
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir

        self.load_data()
        self.load_model()

    def load_data(self, seed=42):
        dataset = load_dataset(self.dataset_name, "Scientific Reasoning - Physics", cache_dir=f"../data/raw/{self.dataset_name}")['train']

        split = dataset.train_test_split(test_size=0.2, seed=seed)
        self.dataset = DatasetDict({
            "train": split['train'],
            "test": split['test']
        })
        print(self.dataset['train'])
        sys.exit()

    def load_model(self):
        self.processor: Qwen3VLProcessor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir = self.cache_dir
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
            cache_dir=self.cache_dir
        )
        self.tokenizer = self.processor.tokenizer

    def split_outside_and_get_numbers(self, data: str):
        pattern = r"<image_start>\[(problem|reasoning)_image_(\d+)\]<image_end>"

        result = []
        pos = 0

        for match in re.finditer(pattern, data):
            start, end = match.span()
            image_num = int(match.group(2))

            chunk = data[pos:start].strip()
            if chunk:
                result.append(("text", chunk))

            result.append(("image", image_num))

            pos = end

        remaining = data[pos:].strip()
        if remaining:
            result.append(("text", remaining))

        return result
    
    def build_prompt(self, question: str, reasoning_step: str, question_imgs: List[Image], reasoning_imgs: List[Image]):
        system_prompt = (
            "You are labeling reasoning steps in visual chains of thought.\n\n"
            f"Your allowed labels are:\n{self.taxonomy}\n\n"
            "Given a question and an associated list of reasoning steps, label EACH step with EXACTLY ONE label.\n"
            "Return a JSON array with objects of the form:\n"
            "[\n"
            "{{'step': 1, 'label': '<label_for_step_1>'}},\n"
            "{{'step': 2, 'label': '<label_for_step_2>'}},\n"
            "..."
            "\n]"
        )

        messages = []

        messages.append({
            "role": "system",
            "content": [{
                "type": "text",
                "text": system_prompt
            }]
        })

        user_content = []

        question_split = self.split_outside_and_get_numbers(question)
        reasoning_split = self.split_outside_and_get_numbers(reasoning_step)

        user_content.append({
            "type": "text",
            "text": "Question"
        })

        for data in question_split:
            if data[0] == "text":
                user_content.append({
                    "type": "text",
                    "text": data[1]
                })
            if data[0] == "image":
                user_content.append({
                    "type": "image",
                    "image": question_imgs[data[1] - 1]
                })
        
        user_content.append({
            "type": "text",
            "text": "\n\nReasoning Steps\n"
        })

        for data in reasoning_split:
            if data[0] == "text":
                user_content.append({
                    "type": "text",
                    "text": data[1]
                })
            if data[0] == "image":
                user_content.append({
                    "type": "image",
                    "image": reasoning_imgs[data[1] - 1]
                })

        messages.append({
            "role": "user",
            "content": user_content
        })

        return messages
    
    def annotate_sample(self, sample):
        message = self.build_prompt(
                    sample['Question'],
                    sample['Text Reasoning Trace'],
                    question_imgs=[sample[f'problem_image_{j + 1}'] for j in range(2)],
                    reasoning_imgs=[sample[f'reasoning_image_{j + 1}'] for j in range(4)]
                )
        
        inputs = self.processor.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode(), torch.amp.autocast(device_type=self.model.device.type):
            generated = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                use_cache=True
            )

        out = self.processor.decode(generated[0], skip_special_tokens=True)
        out_messages = out.split("assistant")
        out = out_messages[-1]
        try:
            parsed = json.loads(out)
            thought_labels = [step['label'] for step in parsed]
        except Exception as e:
            print("Raw output", out)
            thought_labels = ['unparseable']
            

        del inputs, generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return thought_labels
    
    def annotate_dataset(self, output_file="../data/thought_labels.parquet", 
                        save_interval=96, start_idx = 0, end_idx = None):
        
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        all_results = []
        
        dataset_size = len(self.dataset['train'])
        if end_idx is None:
            end_idx = dataset_size
        end_idx = min(dataset_size, end_idx)
        
        pbar = tqdm(
            range(start_idx, end_idx), 
            desc=f"Annotating {start_idx}-{end_idx}",
            initial=start_idx,
            total=end_idx - start_idx
        )

        for idx in pbar:
            sample = self.dataset['train'][idx]
            
            thought_labels = self.annotate_sample(sample)
            
            all_results.append({
                "sample_index": idx,
                "thought_labels": thought_labels
            })

            if len(all_results) % save_interval == 0:
                df = pl.DataFrame(all_results)
                df.write_parquet(output_file)
                pbar.set_postfix({"last_saved": idx})
                
                # Periodic garbage collection
                gc.collect()

        df = pl.DataFrame(all_results)
        df.write_parquet(output_file)
        gc.collect()

        print(f"Annotation complete. Total samples: {len(all_results)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dataset_name", default="multimodal-reasoning-lab/Zebra-CoT")
    parser.add_argument("--cache_dir", default=CACHE_DIR)
    parser.add_argument("--output_prefix", default="../data/thought_labels")
    parser.add_argument("--save_interval", type=int, default=96)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1)

    args = parser.parse_args()

    lt = LabelThoughts(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir
    )
    lt.annotate_dataset(
        output_file=f"{args.output_prefix}.parquet",
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()


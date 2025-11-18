import json
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from datasets import load_dataset
import os
import sys
import polars as pl
import re
from tqdm import tqdm
import gc

from PIL.Image import Image
from typing import List

CACHE_DIR = "/fs/nexus-scratch/adas1236/.cache/huggingface"
torch.backends.cudnn.benchmark = True

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
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

    def load_data(self):
        self.dataset = load_dataset(self.dataset_name, "Scientific Reasoning - Physics", cache_dir=f"../data/raw/{self.dataset_name}")

    def load_model(self):
        self.processor: Qwen3VLProcessor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir = self.cache_dir
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
            cache_dir=self.cache_dir
        )
        self.tokenizer = self.processor.tokenizer
        self.model.eval()

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
    
    def annotate_batch(self, batch, indices):
        messages_batch = []
        for i in range(len(indices)):
            messages_batch.append(
                self.build_prompt(
                    batch['Question'][i],
                    batch['Text Reasoning Trace'][i],
                    question_imgs=[batch[f'problem_image_{j + 1}'][i] for j in range(2)],
                    reasoning_imgs=[batch[f'reasoning_image_{j + 1}'][i] for j in range(4)]
                )
            )
        
        inputs = self.processor.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode(), torch.amp.autocast(self.model.device.type):
            generated = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                use_cache=True
            )

        batch_labels = []

        for i, res in enumerate(generated):
            out = self.processor.decode(res, skip_special_tokens=True)
            out_messages = out.split("assistant")
            out = out_messages[-1]
            try:
                parsed = json.loads(out)
                thought_labels = [step['label'] for step in parsed]
            except Exception as e:
                print("Raw output", out)
                thought_labels = ['unparseable']
            
            batch_labels.append(thought_labels)

        del inputs, generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return batch_labels
    
    def annotate_dataset(self, output_file="../data/thought_labels.parquet", 
                        batch_size=4, save_interval=96):
        
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        if os.path.exists(output_file):
            df_existing = pl.scan_parquet(output_file).select(['sample_index', 'thought_labels'])
            start_idx = df_existing.select(pl.col('sample_index').max()).collect().item() + 1
            all_results = df_existing.collect().to_dicts()
            print(f"Resuming from sample {start_idx}")
        else:
            start_idx = 0
            all_results = []

        dataset_size = len(self.dataset['train'])
        
        pbar = tqdm(
            range(start_idx, dataset_size, batch_size), 
            desc="Annotating",
            initial=start_idx // batch_size,
            total=(dataset_size + batch_size - 1) // batch_size
        )

        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, dataset_size)
            actual_batch_size = batch_end - batch_start
            
            batch_samples = self.dataset['train'][batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            batch_results = self.annotate_batch(batch_samples, indices=batch_indices)
            
            for i, thought_labels in enumerate(batch_results):
                all_results.append({
                    "sample_index": batch_start + i,
                    "thought_labels": thought_labels
                })

            if (len(all_results) - start_idx) % save_interval < actual_batch_size or batch_end >= dataset_size:
                df = pl.DataFrame(all_results)
                df.write_parquet(output_file)
                pbar.set_postfix({"last_saved": batch_end})
                
                # Periodic garbage collection
                gc.collect()

        print(f"Annotation complete. Total samples: {len(all_results)}")


def main():
    lt = LabelThoughts()
    lt.annotate_dataset()

if __name__ == '__main__':
    main()



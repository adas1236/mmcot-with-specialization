import torch
import os, sys

from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model
import wandb
import time

import logging
logging.basicConfig(filename="../logs/collate_errors.log", level=logging.WARNING)

os.environ["HF_HOME"] = "/fs/nexus-scratch/adas1236/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

class SFT_with_LoRA:
    def __init__(self, dataset_name: str, model_name: str):
        self.dataset_name = dataset_name
        self.dataset = load_from_disk(f"../data/processed/{self.dataset_name}")
        self.model_name = model_name

    def load_model(self):
        processor = AutoProcessor.from_pretrained(self.model_name, use_fast = True)
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config = bnb_config,
            device_map="auto",
        )

        return model, processor

    
    def collate_fn(self, batch, processor: AutoProcessor, max_length: int):
        prompts = [sample["prompt"] for sample in batch]
        targets = [sample["target"] for sample in batch]
        images = [[sample['image']] for sample in batch]

        encoded = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        )

        labels = processor.tokenizer(
            targets,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        pad_token_id = processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100

        input_len = encoded["input_ids"].shape[1]
        label_len = labels.shape[1]

        if label_len < input_len:
            pad = torch.full(
                (labels.shape[0], input_len - label_len),
                fill_value=-100,
                dtype=labels.dtype,
            )
            labels = torch.cat([labels, pad], dim=1)
        elif label_len > input_len:
            labels = labels[:, :input_len]

        encoded["labels"] = labels

        return encoded
    
    def train(self):
        model, processor = self.load_model()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        wandb.init(
            project=f"{self.model_name.replace('/', '-')}-lora-sft",
            name=f"{self.model_name.replace('/', '-')}-SFT-{time.time()}",
            save_code=True,
            mode="offline"
        )

        training_args = SFTConfig(
            output_dir=f"../models/{self.model_name}-finetuned",
            logging_dir=f"../logs/tensorboard",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            save_strategy="epoch",
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=1000,
            logging_steps=10,
            logging_first_step=True,
            learning_rate=1e-4,
            max_grad_norm=1.0,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            report_to=["tensorboard", "wandb"],
            bf16=False,
            fp16=True,
            tf32=True,
            hub_private_repo=False,
            push_to_hub=False,
            num_train_epochs=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            packing=False,
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=4
        )

        peft_config = LoraConfig(r=16,
                                lora_alpha=64,
                                lora_dropout=0.05,
                                target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
                                task_type=TaskType.CAUSAL_LM)
        
        model = get_peft_model(model, peft_config)
        
        tokenizer = processor.tokenizer
        max_len = min(getattr(tokenizer, "model_max_length", 2048), 2048)
        
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            data_collator=lambda b: self.collate_fn(b, processor, max_length = max_len),
        )

        trainer.train()
        trainer.save_model()

        valid_metrics = trainer.evaluate(self.dataset["validation"])
        print("Valid metrics:", valid_metrics)
        wandb.log({"valid": valid_metrics})

        test_metrics = trainer.evaluate(self.dataset["test"])
        print("Test metrics:", test_metrics)
        wandb.log({"test": test_metrics})

if __name__ == "__main__":
    trainer = SFT_with_LoRA(
        dataset_name="derek-thomas/ScienceQA",
        model_name="google/gemma-3-4b-it",
    )
    trainer.train()
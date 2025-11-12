from datasets import load_from_disk
from code.preprocess import PreprocessDataset
from code.finetune import SFT_with_LoRA

def main():
    DATASET_NAME = "derek-thomas/ScienceQA"
    pd = PreprocessDataset(dataset_name=DATASET_NAME)
    pd.preprocess_dataset()

if __name__ == "__main__":
    main()

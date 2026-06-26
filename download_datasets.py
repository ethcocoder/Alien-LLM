import os
import requests
import pandas as pd
import json

def download_file(url, filename):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {filename}")

def main():
    # Create datasets directory
    os.makedirs('datasets', exist_ok=True)
    os.chdir('datasets')

    # URLs
    tinystories_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
    alpaca_parquet_url = "https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"

    # Download TinyStories
    if not os.path.exists('TinyStories-valid.txt'):
        download_file(tinystories_url, 'TinyStories-valid.txt')
    else:
        print("TinyStories-valid.txt already exists.")

    # Download Alpaca Parquet
    parquet_file = 'alpaca_train.parquet'
    if not os.path.exists(parquet_file):
        download_file(alpaca_parquet_url, parquet_file)
    else:
        print(f"{parquet_file} already exists.")

    # Convert Parquet to JSONL
    print("Converting Parquet to JSONL...")
    try:
        df = pd.read_parquet(parquet_file)
        with open('alpaca_data.jsonl', 'w') as f:
            for _, row in df.iterrows():
                instruction = row['instruction']
                input_text = row['input']
                output = row['output']
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: "
                f.write(json.dumps({"prompt": prompt, "completion": output}) + '\n')
        print("Conversion complete: alpaca_data.jsonl")
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("Please ensure 'pyarrow' or 'fastparquet' is installed: pip install pyarrow")

if __name__ == "__main__":
    main()

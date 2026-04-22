import json
import os
from datasets import load_dataset

def shrink_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    output_path = os.path.join(data_dir, 'unified_aviation.json')
    
    print("fetching high-stakes narratives from hugging face =>")
    dataset = load_dataset("Timilehin674/Aviation_QA", split="train", field="data")
    
    unified_data = []
    # TARGET: ~2,500 QA pairs usually equals ~10MB of text
    pair_limit = 2500 
    current_count = 0

    for category in dataset:
        if current_count >= pair_limit:
            break
            
        cat_name = category.get('event_category', 'General')
        for p in category['paragraphs']:
            for qa in p['qas']:
                if current_count < pair_limit:
                    unified_data.append({
                        "example_id": qa['id'],
                        "question": qa['question'],
                        "context": p['context'],
                        "gold_answer": qa['answers'][0]['text'],
                        "nodes": [], 
                        "metadata": {"category": cat_name}
                    })
                    current_count += 1

    os.makedirs(data_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(unified_data, f, indent=4)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"SUCCESS => Created a {size_mb:.2f} MB file with {len(unified_data)} records =>")

if __name__ == "__main__":
    shrink_dataset()
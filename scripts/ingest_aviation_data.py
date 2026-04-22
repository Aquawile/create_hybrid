import json
import os
from datasets import load_dataset

def force_fast_ingest():
    # 1 => setup paths relative to the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    output_path = os.path.join(data_dir, 'unified_aviation.json')
    
    # 2 => ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)

    print("fetching a small slice of aviation data from hugging face =>")
    # field="data" ensures we get the NTSB/ASRS narratives
    dataset = load_dataset("Timilehin674/Aviation_QA", split="train", field="data")
    
    unified_data = []
    # 3 => limit to 10 event categories for a lightning-fast file
    # this will give you about 200-500 high-quality QA pairs
    limit = 10 

    for i in range(min(len(dataset), limit)):
        category_data = dataset[i]
        cat_name = category_data.get('event_category', 'General')
        
        for p in category_data['paragraphs']:
            for qa in p['qas']:
                unified_data.append({
                    "example_id": qa['id'],
                    "question": qa['question'],
                    "context": p['context'],
                    "gold_answer": qa['answers'][0]['text'],
                    "nodes": [], # placeholder for your graph layer
                    "metadata": {"category": cat_name}
                })

    # 4 => overwrite the old monster file with this lean version
    with open(output_path, 'w') as f:
        json.dump(unified_data, f, indent=4)
    
    print(f"SUCCESS => Created {output_path} with {len(unified_data)} records =>")
    print("this file should now be visible in your VS Code sidebar =>")

if __name__ == "__main__":
    force_fast_ingest()
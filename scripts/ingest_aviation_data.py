import json
import os
from datasets import load_dataset

def download_and_format_data():
    print("fetching high-stakes aviation data from hugging face =>")
    
    # field="data" returns the list of event categories
    dataset = load_dataset("Timilehin674/Aviation_QA", split="train", field="data")
    
    # use the actual size of the dataset to avoid IndexErrors
    num_categories = len(dataset)
    print(f"found {num_categories} event categories => processing all narratives =>")

    new_queries = {}
    new_gt = {}
    corpus_entries = []
    all_answers = set()

    for i in range(num_categories):
        category_data = dataset[i]
        category_name = category_data.get('event_category', 'Unknown')
        
        # each category has a list of paragraphs (narratives)
        for p_idx, paragraph in enumerate(category_data['paragraphs']):
            context = paragraph['context']
            
            # create a document for the corpus
            doc_id = f"doc_{i}_{p_idx}"
            corpus_entries.append({
                "id": doc_id,
                "text": context,
                "metadata": {"category": category_name}
            })

            # each paragraph has multiple Q&A pairs
            for q_idx, qa in enumerate(paragraph['qas']):
                qid = f"aviation_{i}_{p_idx}_{q_idx}"
                answer_text = qa['answers'][0]['text']
                
                new_queries[qid] = qa['question']
                new_gt[qid] = answer_text
                all_answers.add(answer_text)

    # ensure the /data folder exists
    os.makedirs('data', exist_ok=True)
    
    # writing to 'w' mode creates the files if they do not exist
    files_to_create = {
        'data/aviation_queries.json': new_queries,
        'data/aviation_gt.json': new_gt,
        'data/aviation_corpus.json': corpus_entries,
        'data/aviation_hypotheses.json': list(all_answers)
    }

    for path, content in files_to_create.items():
        with open(path, 'w') as f:
            json.dump(content, f, indent=4)
        print(f"successfully stored => {path}")

    print("ingestion complete => engine is now ready for sensitive scenario testing =>")

if __name__ == "__main__":
    download_and_format_data()
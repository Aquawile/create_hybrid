import json
import os
import re
from datasets import load_dataset

def extract_aviation_entities(text):
    """Extract key aviation entities from text for graph nodes"""
    aviation_terms = {
        "fuel system", "landing gear", "engine", "hydraulic system", "electrical system",
        "asrs narrative", "ntsb report", "cockpit", "avionics", "control surfaces",
        "stall warning", "autopilot", "flight control", "emergency system", "thrust reverser",
        "navigation system", "radar", "transponder", "oxygen system", "fire suppression",
        "landing gear", "flaps", "slats", "elevator", "aileron", "rudder", "spoiler",
        "propulsion", "turbine", "combustion chamber", "fuel pump", "hydraulic pump",
        "battery", "generator", "alternator", "pitot tube", "static port", "airspeed indicator"
    }

    found_entities = set()
    lower_text = text.lower()

    for term in aviation_terms:
        if re.search(r'\b' + re.escape(term) + r'\b', lower_text):
            found_entities.add(term)

    return sorted(list(found_entities))

def download_and_format_data():
    print("fetching high-stakes aviation data from hugging face =>")

    # field="data" returns the list of event categories
    dataset = load_dataset("Timilehin674/Aviation_QA", split="train", field="data")

    # use the actual size of the dataset to avoid IndexErrors
    num_categories = len(dataset)
    print(f"found {num_categories} event categories => processing all narratives =>")

    unified_dataset = []
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
                example_id = f"aviation_{i}_{p_idx}_{q_idx}"
                answer_text = qa['answers'][0]['text']
                question = qa['question']

                all_answers.add(answer_text)

                # Extract entities for graph nodes
                context_nodes = extract_aviation_entities(context)
                question_nodes = extract_aviation_entities(question)
                all_nodes = sorted(list(set(context_nodes + question_nodes)))

                # Create unified DiagnosticObject
                diagnostic_object = {
                    "example_id": example_id,
                    "context": context,
                    "question": question,
                    "gold_answer": answer_text,
                    "candidate_pool": [],
                    "graph_metadata": {
                        "nodes": all_nodes,
                        "doc_id": doc_id,
                        "event_category": category_name
                    }
                }

                unified_dataset.append(diagnostic_object)

    # Populate candidate_pool for all records (global pool)
    candidate_pool = list(all_answers)
    for record in unified_dataset:
        record["candidate_pool"] = candidate_pool

    # ensure the /data folder exists
    os.makedirs('data', exist_ok=True)

    # writing to 'w' mode creates the files if they do not exist
    files_to_create = {
        'data/aviation_queries.json': {r["example_id"]: r["question"] for r in unified_dataset},
        'data/aviation_gt.json': {r["example_id"]: r["gold_answer"] for r in unified_dataset},
        'data/aviation_corpus.json': corpus_entries,
        'data/aviation_hypotheses.json': candidate_pool,
        'data/unified_aviation.json': unified_dataset
    }

    for path, content in files_to_create.items():
        with open(path, 'w') as f:
            json.dump(content, f, indent=4)
        print(f"successfully stored => {path}")

    print(f"\nGenerated unified dataset with {len(unified_dataset)} DiagnosticObjects")
    print(f"Average entities per record: {sum(len(r['graph_metadata']['nodes']) for r in unified_dataset)/len(unified_dataset):.1f} nodes")
    print("ingestion complete => engine is now graph-ready =>")

if __name__ == "__main__":
    download_and_format_data()
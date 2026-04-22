import json
import os

# Load unified dataset
with open('data/unified_aviation.json') as f:
    unified = json.load(f)

# Extract corpus - generate doc_ids from context hash since not present
corpus = []
seen_contexts = set()
for i, r in enumerate(unified):
    import hashlib
    doc_id = f"doc_{hashlib.md5(r['context'].encode()).hexdigest()[:8]}"
    if doc_id not in seen_contexts:
        corpus.append({
            "id": doc_id,
            "text": r["context"],
            "metadata": r["metadata"]
        })
        seen_contexts.add(doc_id)

# Extract other files
queries = {r["example_id"]: r["question"] for r in unified}
gt = {r["example_id"]: r["gold_answer"] for r in unified}
hypotheses = list(set(r["gold_answer"] for r in unified))

# Write files
files_to_create = {
    'data/aviation_corpus.json': corpus,
    'data/aviation_queries.json': queries,
    'data/aviation_gt.json': gt,
    'data/aviation_hypotheses.json': hypotheses
}

os.makedirs('data', exist_ok=True)
for path, content in files_to_create.items():
    with open(path, 'w') as f:
        json.dump(content, f, indent=4)
    print(f"Generated {path}")

print("All files generated from unified dataset")
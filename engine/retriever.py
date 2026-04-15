import json

class DiagnosticRetriever:
    def __init__(self, corpus_path):
        with open(corpus_path, 'r') as f:
            self.corpus = json.load(f)

    def retrieve(self, query, hybrid=False, top_k=2):
        if not hybrid:
            # Baseline: We only give it the first two docs (usually wrong)
            return self.corpus[:top_k]
        
        # Hybrid: Look for the specific document that contains the answer
        # We search the corpus for documents that contain keywords from the query
        keywords = query.lower().replace("'", "").split()
        matches = []
        for doc in self.corpus:
            # If the doc text shares 2 or more words with the query, it's a hit
            common_words = set(keywords).intersection(set(doc['text'].lower().split()))
            if len(common_words) >= 1:
                matches.append(doc)
        
        # If we found relevant docs, return them. Otherwise, return the first few.
        return matches[:top_k] if matches else self.corpus[:top_k]
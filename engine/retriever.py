import json
import math
from collections import Counter


# Common English stopwords to filter out during keyword matching
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
    'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
    'his', 'its', 'our', 'their', 'what', 'which', 'who', 'whom',
    'and', 'but', 'or', 'so', 'yet', 'nor', 'not', 'no', 'do', 'does',
    'did', 'have', 'has', 'had', 'will', 'would', 'could', 'should',
    'may', 'might', 'can', 'into', 'up', 'out', 'down', 'about',
    'over', 'under', 'during', 'before', 'after', 'above', 'below',
}


class DiagnosticRetriever:
    def __init__(self, corpus_path):
        with open(corpus_path, 'r') as f:
            self.corpus = json.load(f)

        # Build TF-IDF index at init time
        self._index_documents()

    def _index_documents(self):
        """Build TF-IDF vectors for all corpus documents."""
        doc_tokens = []
        for doc in self.corpus:
            tokens = self._tokenize(doc['text'])
            doc_tokens.append(tokens)

        num_docs = len(doc_tokens)
        df = Counter()
        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1

        self.vocab = sorted(df.keys())
        self.idf = {}
        for term in self.vocab:
            self.idf[term] = math.log((num_docs + 1) / (df[term] + 1)) + 1

        self.doc_vectors = []
        for tokens in doc_tokens:
            tf = Counter(tokens)
            doc_len = len(tokens)
            vector = {}
            for term in self.vocab:
                if term in tf:
                    tf_norm = tf[term] / doc_len
                    vector[term] = tf_norm * self.idf[term]
                else:
                    vector[term] = 0.0
            self.doc_vectors.append(vector)

    def _tokenize(self, text):
        """Simple whitespace tokenizer with lowercase, stopword removal."""
        raw = text.lower().replace("'", "").split()
        return [w for w in raw if w not in STOPWORDS and len(w) > 2]

    def _compute_tfidf_vector(self, tokens):
        """Compute TF-IDF vector for a list of tokens."""
        tf = Counter(tokens)
        doc_len = len(tokens)
        vector = {}
        for term in self.vocab:
            if term in tf:
                tf_norm = tf[term] / doc_len
                vector[term] = tf_norm * self.idf[term]
            else:
                vector[term] = 0.0
        return vector

    def _cosine_similarity(self, vec_a, vec_b):
        """Compute cosine similarity between two sparse vectors."""
        dot = sum(vec_a.get(term, 0) * vec_b.get(term, 0) for term in self.vocab)
        norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query, hybrid=False, top_k=2):
        """
        Retrieve documents for a query.

        Baseline mode (hybrid=False):
            No retrieval intelligence — returns documents sorted by ID (D1, D2…).
            This simulates a system with no search capability at all,
            just serving documents in fixed order. Represents the "no RAG"
            case: the LLM gets whatever docs happen to be first in the database,
            which are almost never relevant to the specific query.

        Hybrid mode (hybrid=True):
            TF-IDF cosine similarity — proper information retrieval that
            weights terms by how discriminative they are across the corpus.
            Rare, diagnostic-specific terms get higher weight while common
            terms get suppressed. Returns documents that are truly relevant.
        """
        if not hybrid:
            # Baseline: fixed document order — no search, just sequential
            # This is the "no retrieval intelligence" baseline
            return self.corpus[:top_k]

        # Hybrid: TF-IDF cosine similarity (proper IR)
        query_tokens = self._tokenize(query)
        query_vector = self._compute_tfidf_vector(query_tokens)

        scores = []
        for i, doc_vector in enumerate(self.doc_vectors):
            score = self._cosine_similarity(query_vector, doc_vector)
            scores.append((score, i))

        scores.sort(key=lambda x: (-x[0], x[1]))
        return [self.corpus[i] for score, i in scores[:top_k]]

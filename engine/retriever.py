import json
import math
import re
from collections import Counter, defaultdict


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

    def _extract_concept_nodes(self, text):
        """Extract aviation concept nodes from text for graph traversal"""
        aviation_terms = {
            "fuel system", "landing gear", "engine", "hydraulic system", "electrical system",
            "asrs narrative", "ntsb report", "cockpit", "avionics", "control surfaces",
            "stall warning", "autopilot", "flight control", "emergency system", "thrust reverser",
            "navigation system", "radar", "transponder", "oxygen system", "fire suppression",
            "flaps", "slats", "elevator", "aileron", "rudder", "spoiler",
            "propulsion", "turbine", "combustion chamber", "fuel pump", "hydraulic pump",
            "battery", "generator", "alternator", "pitot tube", "static port", "airspeed indicator"
        }
        
        found_nodes = set()
        lower_text = text.lower()
        
        for term in aviation_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', lower_text):
                found_nodes.add(term)
        
        return sorted(list(found_nodes))


    def _normalize_concept(self, concept):
        """Normalize concept names for better matching"""
        # Remove article prefixes
        concept = re.sub(r'^(the|a|an)\s+', '', concept.lower().strip())
        # Standardize terms
        replacements = {
            'system': '_sys',
            'failure': '_fail',
            'malfunction': '_fail', 
            'problem': '_issue',
            'issue': '_issue',
            'error': '_err',
            'fault': '_err'
        }
        for old, new in replacements.items():
            concept = re.sub(r'\b' + old + r'\b', new, concept)
        return concept

    def _extract_concept_nodes_enhanced(self, text):
        """Enhanced entity extraction with normalization"""
        base_entities = self._extract_concept_nodes(text)
        normalized = [self._normalize_concept(e) for e in base_entities]
        return sorted(set(normalized + base_entities))

    def _compute_path_score(self, query_nodes, doc_nodes):
        """Score document by weighted path length to query concepts"""
        if not query_nodes or not doc_nodes:
            return 0.0
        
        # Direct overlap score
        direct_overlap = len(set(query_nodes) & set(doc_nodes))
        
        # Path score: penalize by distance (simplified as inverse overlap)
        if direct_overlap > 0:
            path_score = direct_overlap / (1 + (len(query_nodes) - direct_overlap))
        else:
            path_score = 0.0
        
        return direct_overlap * 0.7 + path_score * 0.3

    def retrieve_subgraph_enhanced(self, query, top_k=3):
        """Enhanced subgraph retrieval with normalization and path scoring"""
        query_nodes = self._extract_concept_nodes_enhanced(query)
        
        if not query_nodes:
            return self.retrieve(query, hybrid=True, top_k=top_k)
        
        # Build document concept index
        if not hasattr(self, 'concept_index_enhanced'):
            self.concept_index_enhanced = defaultdict(list)
            for idx, doc in enumerate(self.corpus):
                doc_nodes = self._extract_concept_nodes_enhanced(doc['text'])
                for node in doc_nodes:
                    self.concept_index_enhanced[node].append(idx)
        
        # Find neighborhood
        neighborhood_docs = set()
        for node in query_nodes:
            if node in self.concept_index_enhanced:
                neighborhood_docs.update(self.concept_index_enhanced[node])
        
        # Score by path strength
        doc_scores = []
        for doc_idx in neighborhood_docs:
            doc = self.corpus[doc_idx]
            doc_nodes = self._extract_concept_nodes_enhanced(doc['text'])
            path_score = self._compute_path_score(query_nodes, doc_nodes)
            doc_scores.append((path_score, doc_idx))
        
        doc_scores.sort(key=lambda x: (-x[0], x[1]))
        return [self.corpus[idx] for _, idx in doc_scores[:top_k]]

    def retrieve_subgraph(self, query, top_k=3):
        """
        Graph-based subgraph retrieval:
        1. Extract concept nodes from query
        2. Traverse 1-hop neighborhood from matched concepts
        3. Retrieve documents connected to matched nodes
        4. Score by connectivity strength
        """
        query_nodes = self._extract_concept_nodes(query)
        
        if not query_nodes:
            # Fall back to standard hybrid retrieval if no concepts matched
            return self.retrieve(query, hybrid=True, top_k=top_k)
        
        # Build document concept index on first use
        if not hasattr(self, 'concept_index'):
            self.concept_index = defaultdict(list)
            for idx, doc in enumerate(self.corpus):
                doc_nodes = self._extract_concept_nodes(doc['text'])
                for node in doc_nodes:
                    self.concept_index[node].append(idx)
        
        # Get all documents in query node neighborhood
        neighborhood_docs = set()
        for node in query_nodes:
            if node in self.concept_index:
                neighborhood_docs.update(self.concept_index[node])
        
        # Score documents by number of matching concepts
        doc_scores = []
        for doc_idx in neighborhood_docs:
            doc = self.corpus[doc_idx]
            doc_nodes = self._extract_concept_nodes(doc['text'])
            overlap = len(set(query_nodes) & set(doc_nodes))
            doc_scores.append((overlap, doc_idx))
        
        # Sort by highest concept overlap first
        doc_scores.sort(key=lambda x: (-x[0], x[1]))
        
        return [self.corpus[idx] for _, idx in doc_scores[:top_k]]

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

    def compute_graph_coverage(self, query, hypotheses, gold_answer):
        """
        Compute coverage metrics for graph retrieval:
        - Entity overlap between query, gold answer, and retrieved subgraph
        - Term overlap scores
        - Returns dict with coverage stats
        """
        query_nodes = self._extract_concept_nodes(query)
        gold_nodes = self._extract_concept_nodes(gold_answer)

        # Retrieve subgraph
        subgraph_docs = self.retrieve_subgraph(query, top_k=3)
        subgraph_text = " ".join([doc['text'] for doc in subgraph_docs]).lower()
        subgraph_nodes = self._extract_concept_nodes(subgraph_text)

        # Entity coverage
        query_gold_overlap = len(set(query_nodes) & set(gold_nodes)) / max(1, len(set(query_nodes) | set(gold_nodes)))
        subgraph_gold_overlap = len(set(subgraph_nodes) & set(gold_nodes)) / max(1, len(set(subgraph_nodes) | set(gold_nodes)))

        # Term overlap with gold answer
        gold_terms = set(gold_answer.lower().split())
        subgraph_terms = set(subgraph_text.split())
        term_overlap = len(gold_terms & subgraph_terms) / max(1, len(gold_terms))

        # Hypothesis discrimination: how many hypotheses have support in subgraph
        hyp_coverage = {}
        for hyp in hypotheses:
            hyp_terms = set(hyp.lower().split())
            hyp_overlap = len(hyp_terms & subgraph_terms) / max(1, len(hyp_terms))
            hyp_coverage[hyp] = hyp_overlap

        return {
            "query_nodes": query_nodes,
            "gold_nodes": gold_nodes,
            "subgraph_nodes": subgraph_nodes,
            "query_gold_entity_overlap": query_gold_overlap,
            "subgraph_gold_entity_overlap": subgraph_gold_overlap,
            "term_overlap": term_overlap,
            "hypothesis_coverage": hyp_coverage,
            "gold_coverage": hyp_coverage.get(gold_answer, 0),
            "max_distractor_coverage": max([v for k, v in hyp_coverage.items() if k != gold_answer] or [0]),
            "coverage_ratio": hyp_coverage.get(gold_answer, 0) / max(0.01, max([v for k, v in hyp_coverage.items() if k != gold_answer] or [0]))
        }

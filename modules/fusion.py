import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from config import FUSION_SIMILARITY_THRESHOLD

class KnowledgeFusion:
    """
    Module 2: Knowledge Fusion
    - ATOM citation: Cosine similarity-based Entity Resolution
    - EDC citation: Apply Canonicalize step
    - Speed optimization through pre-computed embeddings
    """

    def __init__(self, similarity_threshold=FUSION_SIMILARITY_THRESHOLD,
                 model_name='all-MiniLM-L6-v2'):
        self.similarity_threshold = similarity_threshold
        print("Loading embedding model...", flush=True)
        self.embed_model = SentenceTransformer(model_name)
        print("Embedding model loaded", flush=True)

    def normalize(self, text):
        """Text normalization"""
        if not text:
            return ""
        return str(text).lower().strip()

    def batch_embeddings(self, texts):
        """
        Batch embedding calculation
        Embed all texts at once → Significant speed improvement
        """
        if not texts:
            return np.array([])
        return self.embed_model.encode(texts, show_progress_bar=False)

    def cosine_sim_matrix(self, embeddings):
        """
        Calculate cosine similarity matrix at once
        O(n²) individual calculations → O(1) matrix operation
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms
        return np.dot(normalized, normalized.T)

    def canonicalize_entities(self, quintuples):
        """
        EDC citation - Canonicalize step
        Merge entities with the same meaning into representative form
        """
        all_entities = list(set(
            [q.get('entity1', '') for q in quintuples] +
            [q.get('entity2', '') for q in quintuples]
        ))
        all_entities = [e for e in all_entities if e]

        if len(all_entities) < 2:
            return quintuples

        # Calculate batch embeddings
        embeddings = self.batch_embeddings(all_entities)
        sim_matrix = self.cosine_sim_matrix(embeddings)

        # Group similar entities (θ = 0.9)
        groups = []
        used = [False] * len(all_entities)

        for i in range(len(all_entities)):
            if used[i]:
                continue
            group = [all_entities[i]]
            used[i] = True

            for j in range(i + 1, len(all_entities)):
                if used[j]:
                    continue
                if sim_matrix[i][j] >= 0.9:
                    group.append(all_entities[j])
                    used[j] = True

            groups.append(group)

        # Select representative form (longest name)
        canonical_map = {}
        for group in groups:
            if len(group) > 1:
                canonical = max(group, key=len)
                for entity in group:
                    canonical_map[entity] = canonical

        # Apply to quintuples
        canonicalized = []
        for q in quintuples:
            q_copy = q.copy()
            e1 = q_copy.get('entity1', '')
            e2 = q_copy.get('entity2', '')

            if e1 in canonical_map:
                q_copy['entity1_original'] = e1
                q_copy['entity1'] = canonical_map[e1]
            if e2 in canonical_map:
                q_copy['entity2_original'] = e2
                q_copy['entity2'] = canonical_map[e2]

            canonicalized.append(q_copy)

        if canonical_map:
            print(f"  Canonicalize: {len(canonical_map)} "
                  f"entities normalized", flush=True)

        return canonicalized

    def deduplicate(self, quintuples):
        """
        Remove duplicate quintuples
        Speed optimization through batch embeddings
        """
        if not quintuples:
            return []

        # Convert each quintuple to a single string
        quintuple_texts = []
        for q in quintuples:
            text = (
                f"{q.get('entity1', '')} "
                f"{q.get('relation', '')} "
                f"{q.get('entity2', '')} "
                f"{q.get('time', '')}"
            )
            quintuple_texts.append(text)

        # Batch embeddings + similarity matrix
        print("  Calculating embeddings...", flush=True)
        embeddings = self.batch_embeddings(quintuple_texts)
        sim_matrix = self.cosine_sim_matrix(embeddings)
        print("  Comparing similarities...", flush=True)

        # Grouping
        groups = []
        used = [False] * len(quintuples)

        for i in range(len(quintuples)):
            if used[i]:
                continue
            group = [quintuples[i]]
            used[i] = True

            for j in range(i + 1, len(quintuples)):
                if used[j]:
                    continue
                if sim_matrix[i][j] >= self.similarity_threshold:
                    group.append(quintuples[j])
                    used[j] = True

            groups.append(group)

        # Resolve conflicts
        fused = []
        for group in groups:
            resolved = self.resolve_conflict(group)
            if resolved:
                fused.append(resolved)

        return fused

    def resolve_conflict(self, candidates):
        """
        Select representative from conflicting quintuples
        Adopt the most frequently appearing location
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        location_count = defaultdict(int)
        for q in candidates:
            loc = self.normalize(q.get('location', 'unknown'))
            location_count[loc] += 1

        best_location = max(location_count, key=location_count.get)

        result = candidates[0].copy()
        for q in candidates:
            if self.normalize(q.get('location', '')) == best_location:
                result = q.copy()
                break

        result['location'] = best_location
        result['merge_count'] = len(candidates)
        return result

    def fuse(self, quintuples):
        """
        Complete fusion pipeline
        Step 1: Canonicalize (EDC)
        Step 2: Deduplicate (ATOM)
        """
        if not quintuples:
            return []

        print("  Step 1: Entity Canonicalize (EDC)...", flush=True)
        canonicalized = self.canonicalize_entities(quintuples)

        print("  Step 2: Deduplicate (ATOM)...", flush=True)
        fused = self.deduplicate(canonicalized)

        return fused

    def get_stats(self, before, after):
        """Return fusion statistics"""
        return {
            "before": len(before),
            "after": len(after),
            "removed": len(before) - len(after),
            "reduction_rate": (len(before) - len(after)) / len(before) * 100
            if len(before) > 0 else 0
        }
"""
Module 1 Postprocessing: Entity + Relation + Location Normalization
LLM-based + Rule-based correction
"""

import ollama
from config import LLM_MODEL, LLM_HOST

# ============================================================
# Rule-based Entity2 correction mapping
# ============================================================
ENTITY2_CORRECTIONS = {
    # Product → Company
    "chatgpt": "OpenAI",
    "gpt-4": "OpenAI",
    "gpt-3": "OpenAI",
    "dall-e": "OpenAI",
    "gemini": "Google",
    "bard": "Google",
    "vision pro": "Apple",
    "iphone": "Apple",
    "twitter": "Twitter",
    "starship": "SpaceX",
    "falcon 9": "SpaceX",
    "ai act": "European Union",

    # Group → Country
    "americans": "United States",
    "american citizens": "United States",
    "russians": "Russia",
    "chinese citizens": "China",
    "ukrainians": "Ukraine",
    "iranians": "Iran",
    "palestinians": "Palestine",
    "palestinianrefugees": "Palestine",
    "palestinian refugees": "Palestine",
    "israelis": "Israel",
    "employees": "Employees",
    "flood victims": "Flood victims",
    "residents": "Residents",
    "mourners": "Protesters",

    # Event → Organization
    "cop27": "United Nations",
    "cop28": "United Nations",
    "cop29": "United Nations",
    "g7": "G7",
    "g20": "G20",
    "brics summit": "BRICS",

    # Region → Country
    "amazon": "Brazil",
    "amazon rainforest": "Brazil",
    "crimea": "Ukraine",
    "donbas": "Ukraine",
    "gaza": "Palestine",
    "west bank": "Palestine",
    "taiwan strait": "Taiwan",
    "south china sea": "China",
    "east sea": "South Korea",
    "british columbia": "Canada",
    "california": "United States",
    "hawaii": "United States",
    "maui": "United States",
    "lahaina": "United States",

    # Organization abbreviations
    "chinese diplomat": "China",
    "russian diplomat": "Russia",
    "acquisition target": "Unknown",
}

# ============================================================
# Rule-based Entity abbreviation/variant unification
# ============================================================
ENTITY_ALIASES = {
    "us": "United States",
    "usa": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "britain": "United Kingdom",
    "eu": "European Union",
    "un": "United Nations",
    "nato": "NATO",
    "uae": "United Arab Emirates",
    "dprk": "North Korea",
    "rok": "South Korea",
    "prc": "China",
    "ecb": "European Central Bank",
    "imf": "International Monetary Fund",
    "icc": "International Criminal Court",
    "who": "World Health Organization",
}


class LocationNormalizer:
    """LLM-based Location normalizer"""

    def __init__(self, model=LLM_MODEL, host=LLM_HOST):
        self.model = model
        self.client = ollama.Client(host=host)
        self.cache = {}

    def normalize(self, location_str):
        if not location_str or location_str.lower() == 'unknown':
            return location_str

        if location_str in self.cache:
            return self.cache[location_str]

        prompt = f"""Given this location string: "{location_str}"

Task: Return ONLY the country or territory name.
- If it is already a country/territory name, return it EXACTLY as is.
- If it is a city, district, or region, return its country/territory.
- PRESERVE the full official name. Do NOT shorten it.
- Return ONLY the name. No explanation.

Examples:
"Washington D.C." → "United States"
"Riyadh, Saudi Arabia" → "Saudi Arabia"
"London, England" → "United Kingdom"
"Brussels" → "Belgium"
"Moscow" → "Russia"
"Beijing" → "China"
"The Hague" → "Netherlands"
"unknown" → "unknown"

Location: "{location_str}"
Country:"""

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response['message']['content'].strip().strip('"\'.,')
            self.cache[location_str] = result
            return result
        except Exception as e:
            return location_str


class RelationNormalizer:
    """LLM-based Relation normalizer"""

    def __init__(self, model=LLM_MODEL, host=LLM_HOST):
        self.model = model
        self.client = ollama.Client(host=host)
        self.cache = {}

    def normalize(self, relation_str):
        if not relation_str or relation_str.lower() == 'unknown':
            return relation_str

        if relation_str in self.cache:
            return self.cache[relation_str]

        prompt = f"""Normalize this relation to a standard knowledge graph action.

Rules:
1. Use base verb form (e.g. "denied" → "Deny")
2. Remove ONLY specific targets and details
3. KEEP the core action phrase (verb + object type)
4. Do NOT reduce to a single verb
5. Capitalize the first word
6. Return ONLY the normalized relation

Examples:
"denied responsibility for oil production cuts" → "Deny responsibility"
"signed a formal agreement with" → "Sign agreement"
"launched ballistic missiles into the East Sea" → "Launch missiles"
"imposed martial law in four regions" → "Impose martial law"
"provided humanitarian aid to flood victims" → "Provide aid"
"Became Prime Minister" → "Become leader"
"Declare completion" → "Declare"
"Order strikes" → "Order military strikes"
"Release" → "Release prisoners"
"Address" → "Request support"
"Broker deal" → "Sign agreement"

Relation: "{relation_str}"
Normalized:"""

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response['message']['content'].strip().strip('"\'.,')
            self.cache[relation_str] = result
            return result
        except Exception as e:
            return relation_str


class EntityNormalizer:
    """LLM-based Entity normalizer - includes high-level mapping"""

    def __init__(self, model=LLM_MODEL, host=LLM_HOST):
        self.model = model
        self.client = ollama.Client(host=host)
        self.cache = {}

    def normalize(self, entity_str):
        if not entity_str or entity_str.lower() == 'unknown':
            return entity_str

        if entity_str in self.cache:
            return self.cache[entity_str]

        # Check rule-based first (save LLM calls)
        lower = entity_str.lower().strip()
        if lower in ENTITY_ALIASES:
            result = ENTITY_ALIASES[lower]
            self.cache[entity_str] = result
            return result

        prompt = f"""Normalize this entity name for a knowledge graph.

Rules:
1. If it is a city, region, or geographic subdivision, return the COUNTRY name
   (e.g. "Mykolaiv" → "Ukraine", "Kyiv" → "Ukraine")
2. If it is a sub-organization or military branch, return the parent COUNTRY
   (e.g. "IRGC Members" → "Iran", "Russian Forces" → "Russia")
3. Remove titles ONLY if a person's name follows
   (e.g. "President Putin" → "Vladimir Putin")
4. If it is already a country or a person's name, keep it as is
5. Remove articles (The, A, An)
6. Return ONLY the normalized name

Examples:
"Ukrainian troops" → "Ukraine"
"Russian Forces" → "Russia"
"IRGC Members" → "Iran"
"Iranian police" → "Iran"
"Israeli Prime Minister Benjamin Netanyahu" → "Benjamin Netanyahu"
"The European Union" → "European Union"
"United Nations General Assembly" → "United Nations"
"Constitutional Court of Russia" → "Russia"
"European Council" → "European Union"
"North Korea" → "North Korea"
"Vladimir Putin" → "Vladimir Putin"

Entity: "{entity_str}"
Normalized:"""

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response['message']['content'].strip().strip('"\'.,')
            self.cache[entity_str] = result
            return result
        except Exception as e:
            return entity_str


def apply_entity2_corrections(entity2_str):
    """Rule-based Entity2 correction (without LLM calls)"""
    if not entity2_str:
        return entity2_str

    lower = entity2_str.lower().strip()

    # 정확 매칭
    if lower in ENTITY2_CORRECTIONS:
        return ENTITY2_CORRECTIONS[lower]

    # 부분 매칭 (포함 관계)
    for key, value in ENTITY2_CORRECTIONS.items():
        if key in lower:
            return value

    return entity2_str


_location_normalizer = None
_relation_normalizer = None
_entity_normalizer = None

def get_location_normalizer():
    global _location_normalizer
    if _location_normalizer is None:
        _location_normalizer = LocationNormalizer()
    return _location_normalizer

def get_relation_normalizer():
    global _relation_normalizer
    if _relation_normalizer is None:
        _relation_normalizer = RelationNormalizer()
    return _relation_normalizer

def get_entity_normalizer():
    global _entity_normalizer
    if _entity_normalizer is None:
        _entity_normalizer = EntityNormalizer()
    return _entity_normalizer

def postprocess_quintuples(quintuples, normalize_relation=True,
                           normalize_entity=True):
    """
    Postprocess quintuple list
    - Location normalization (LLM)
    - Relation normalization (LLM)
    - Entity normalization (LLM + rules)
    - Entity2 correction (rule-based)
    """
    if not quintuples:
        return []

    loc_norm = get_location_normalizer()
    rel_norm = get_relation_normalizer() if normalize_relation else None
    ent_norm = get_entity_normalizer() if normalize_entity else None

    processed = []
    for q in quintuples:
        if not isinstance(q, dict):
            continue
        q_copy = q.copy()

        # Location normalization
        original_location = q.get('location', 'unknown')
        q_copy['location_original'] = original_location
        q_copy['location'] = loc_norm.normalize(original_location)

        # Relation normalization
        if rel_norm:
            original_relation = q.get('relation', 'unknown')
            q_copy['relation_original'] = original_relation
            q_copy['relation'] = rel_norm.normalize(original_relation)

        # Entity normalization
        if ent_norm:
            original_e1 = q.get('entity1', 'unknown')
            original_e2 = q.get('entity2', 'unknown')
            q_copy['entity1_original'] = original_e1
            q_copy['entity2_original'] = original_e2
            q_copy['entity1'] = ent_norm.normalize(original_e1)
            q_copy['entity2'] = ent_norm.normalize(original_e2)

        # Entity2 rule-based correction (after LLM normalization)
        if normalize_entity:
            q_copy['entity2'] = apply_entity2_corrections(
                q_copy['entity2'])

        processed.append(q_copy)

    return processed
"""
Module 1 Postprocessing: Entity + Relation + Location Normalization
LLM-based + Rule-based correction

v12 changes
- every ollama call uses deterministic options (temperature 0, seed 42)
- normalizer replies are reduced to the first non-empty line, prefixes and
  quotes are stripped, and the original value is kept when the reply is
  empty / too long / contains CJK characters (counted in NORMALIZER_STATS,
  details appended to FALLBACK_LOG)
- few-shot examples replaced with strings that do not occur in the test set
- entity2 rule correction can be switched off, and the value before the rule
  plus the matched rule key are recorded on each quintuple
"""

import ollama
from config import LLM_MODEL, LLM_HOST
from modules.llm_common import LLM_OPTIONS, clean_normalizer_response

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

# ============================================================
# Fallback bookkeeping (normalizer replies that were rejected)
# ============================================================
NORMALIZER_STATS = {
    "location": {"calls": 0, "fallbacks": 0},
    "relation": {"calls": 0, "fallbacks": 0},
    "entity": {"calls": 0, "fallbacks": 0},
}
FALLBACK_LOG = []  # list of dicts: normalizer, original, reason, candidate, raw
ERROR_LOG = []     # list of dicts: normalizer, original, error (LLM call failed)


def drain_fallback_log():
    """Return and clear the fallback log (used per sentence by run_extract)."""
    global FALLBACK_LOG
    items, FALLBACK_LOG = FALLBACK_LOG, []
    return items


def drain_error_log():
    """Return and clear the LLM-error log (a failed call keeps the original
    value; run_extract treats this as a transient failure and retries)."""
    global ERROR_LOG
    items, ERROR_LOG = ERROR_LOG, []
    return items


def get_normalizer_stats():
    return {k: dict(v) for k, v in NORMALIZER_STATS.items()}


class _LLMNormalizer:
    """Shared machinery: cached deterministic call + reply cleaning."""

    kind = "generic"

    def __init__(self, model=LLM_MODEL, host=LLM_HOST):
        self.model = model
        self.client = ollama.Client(host=host)
        self.cache = {}

    def _record_error(self, original, exc):
        ERROR_LOG.append({"normalizer": self.kind, "original": original,
                          "error": str(exc)[:300]})
        print(f"[normalizer:{self.kind}] LLM error for {original!r}: {exc}",
              flush=True)

    def _ask(self, prompt, original):
        NORMALIZER_STATS[self.kind]["calls"] += 1
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options=LLM_OPTIONS,
        )
        raw = response['message']['content']
        value, info = clean_normalizer_response(raw, original)
        if info:
            NORMALIZER_STATS[self.kind]["fallbacks"] += 1
            FALLBACK_LOG.append({
                "normalizer": self.kind,
                "original": original,
                "reason": info["reason"],
                "candidate": info["candidate"][:200],
                "raw": raw[:500],
            })
            print(f"[normalizer:{self.kind}] kept original "
                  f"{original!r} ({info['reason']})", flush=True)
        return value


class LocationNormalizer(_LLMNormalizer):
    """LLM-based Location normalizer"""

    kind = "location"

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
"Lima, Peru" → "Peru"
"Edinburgh, Scotland" → "United Kingdom"
"Vienna" → "Austria"
"Hanoi" → "Vietnam"
"Lagos" → "Nigeria"
"Montevideo" → "Uruguay"
"Chiang Mai" → "Thailand"
"unknown" → "unknown"

Location: "{location_str}"
Country:"""

        try:
            result = self._ask(prompt, location_str)
            self.cache[location_str] = result
            return result
        except Exception as e:
            self._record_error(location_str, e)
            return location_str


class RelationNormalizer(_LLMNormalizer):
    """LLM-based Relation normalizer"""

    kind = "relation"

    def normalize(self, relation_str):
        if not relation_str or relation_str.lower() == 'unknown':
            return relation_str

        if relation_str in self.cache:
            return self.cache[relation_str]

        prompt = f"""Normalize this relation to a standard knowledge graph action.

Rules:
1. Use base verb form (e.g. "rejected" → "Reject")
2. Remove ONLY specific targets and details
3. KEEP the core action phrase (verb + object type)
4. Do NOT reduce to a single verb
5. Capitalize the first word
6. Return ONLY the normalized relation

Examples:
"rejected accusations of election fraud" → "Reject accusations"
"entered into a bilateral trade pact with" → "Sign trade pact"
"fired artillery shells across the Yalu River" → "Fire artillery"
"declared a curfew in three provinces" → "Declare curfew"
"donated medical supplies to earthquake survivors" → "Donate supplies"
"Was sworn in as Chancellor" → "Take office"
"Confirm completion" → "Confirm"
"Recall envoy" → "Recall ambassador"
"Apprehend" → "Apprehend suspects"
"Plead" → "Request clemency"
"Ink accord" → "Sign treaty"

Relation: "{relation_str}"
Normalized:"""

        try:
            result = self._ask(prompt, relation_str)
            self.cache[relation_str] = result
            return result
        except Exception as e:
            self._record_error(relation_str, e)
            return relation_str


class EntityNormalizer(_LLMNormalizer):
    """LLM-based Entity normalizer - includes high-level mapping"""

    kind = "entity"

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
   (e.g. "Marseille" → "France", "Osaka" → "Japan")
2. If it is a sub-organization or military branch, return the parent COUNTRY
   (e.g. "Basij militia members" → "Iran", "Turkish Coast Guard Command" → "Turkey")
3. Remove titles ONLY if a person's name follows
   (e.g. "Chancellor Merkel" → "Angela Merkel")
4. If it is already a country or a person's name, keep it as is
5. Remove articles (The, A, An)
6. Return ONLY the normalized name

Examples:
"Polish border guards" → "Poland"
"Turkish Coast Guard Command" → "Turkey"
"Basij militia members" → "Iran"
"Nigerian police" → "Nigeria"
"Hungarian Prime Minister Viktor Orban" → "Viktor Orban"
"The African Union" → "African Union"
"ASEAN Secretariat" → "ASEAN"
"Supreme Court of Kenya" → "Kenya"
"Arab League Council" → "Arab League"
"Uruguay" → "Uruguay"
"Angela Merkel" → "Angela Merkel"

Entity: "{entity_str}"
Normalized:"""

        try:
            result = self._ask(prompt, entity_str)
            self.cache[entity_str] = result
            return result
        except Exception as e:
            self._record_error(entity_str, e)
            return entity_str


def match_entity2_correction(entity2_str):
    """
    Return (corrected_value, matched_key) for the rule-based Entity2
    correction, or (entity2_str, None) when no rule applies.
    Exact match first, then substring match (dict order).
    """
    if not entity2_str:
        return entity2_str, None

    lower = entity2_str.lower().strip()

    if lower in ENTITY2_CORRECTIONS:
        return ENTITY2_CORRECTIONS[lower], lower

    for key, value in ENTITY2_CORRECTIONS.items():
        if key in lower:
            return value, key

    return entity2_str, None


def apply_entity2_corrections(entity2_str):
    """Rule-based Entity2 correction (without LLM calls)"""
    return match_entity2_correction(entity2_str)[0]


_location_normalizer = None
_relation_normalizer = None
_entity_normalizer = None
_normalizer_model = LLM_MODEL
_normalizer_host = LLM_HOST


def configure_normalizers(model=None, host=None):
    """Override the model/host used by the shared normalizers (must be
    called before the first normalization; resets the singletons)."""
    global _normalizer_model, _normalizer_host
    global _location_normalizer, _relation_normalizer, _entity_normalizer
    if model:
        _normalizer_model = model
    if host:
        _normalizer_host = host
    _location_normalizer = _relation_normalizer = _entity_normalizer = None


def get_location_normalizer():
    global _location_normalizer
    if _location_normalizer is None:
        _location_normalizer = LocationNormalizer(_normalizer_model, _normalizer_host)
    return _location_normalizer

def get_relation_normalizer():
    global _relation_normalizer
    if _relation_normalizer is None:
        _relation_normalizer = RelationNormalizer(_normalizer_model, _normalizer_host)
    return _relation_normalizer

def get_entity_normalizer():
    global _entity_normalizer
    if _entity_normalizer is None:
        _entity_normalizer = EntityNormalizer(_normalizer_model, _normalizer_host)
    return _entity_normalizer

def postprocess_quintuples(quintuples, normalize_relation=True,
                           normalize_entity=True, apply_corrections=True):
    """
    Postprocess quintuple list
    - Location normalization (LLM)
    - Relation normalization (LLM)
    - Entity normalization (LLM + rules)
    - Entity2 correction (rule-based; can be disabled)

    Each processed quintuple additionally carries
      *_original           value before LLM normalization
      entity2_before_rules value before the rule-based correction
      entity2_rule         matched ENTITY2_CORRECTIONS key (or None)
    so the effect of the rules can be re-evaluated without re-running LLMs.
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
        original_location = str(q.get('location', 'unknown'))
        q_copy['location_original'] = original_location
        q_copy['location'] = loc_norm.normalize(original_location)

        # Relation normalization
        if rel_norm:
            original_relation = str(q.get('relation', 'unknown'))
            q_copy['relation_original'] = original_relation
            q_copy['relation'] = rel_norm.normalize(original_relation)

        # Entity normalization
        if ent_norm:
            original_e1 = str(q.get('entity1', 'unknown'))
            original_e2 = str(q.get('entity2', 'unknown'))
            q_copy['entity1_original'] = original_e1
            q_copy['entity2_original'] = original_e2
            q_copy['entity1'] = ent_norm.normalize(original_e1)
            q_copy['entity2'] = ent_norm.normalize(original_e2)

        # Entity2 rule-based correction (after LLM normalization)
        if normalize_entity:
            before = str(q_copy.get('entity2', 'unknown'))
            corrected, key = match_entity2_correction(before)
            q_copy['entity2_before_rules'] = before
            q_copy['entity2_rule'] = key
            if apply_corrections:
                q_copy['entity2'] = corrected

        processed.append(q_copy)

    return processed

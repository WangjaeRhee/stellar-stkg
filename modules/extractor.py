import ollama
import json
from collections import Counter
from config import LLM_MODEL, LLM_HOST
from modules.llm_common import LLM_OPTIONS

class STKGExtractor:
    """
    Module 1: Knowledge Extraction
    Multi-stage Extraction + CoT + Self-Consistency
    Time/Location extracted from original text (prevent information loss)
    """

    def __init__(self, model=LLM_MODEL, host=LLM_HOST):
        self.model = model
        self.client = ollama.Client(host=host)
        # v12: few-shot examples replaced so that none overlaps with the
        # 450 test sentences; deterministic decoding (temperature 0, seed 42)
        self.prompt_version = "v12_clean_examples"
        self.last_error = None
        self.error_count = 0

    def _call_llm(self, prompt):
        """Common function for LLM API calls (deterministic options)"""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options=LLM_OPTIONS,
            )
            return response['message']['content']
        except Exception as e:
            self.last_error = str(e)
            self.error_count += 1
            print(f"LLM error: {e}")
            return ""

    def decompose_to_atomic_facts(self, text):
        """Step 1: Atomic Fact Decomposition"""
        prompt = f"""Decompose the following text into atomic facts.
Each atomic fact should contain exactly ONE event with:
- ONE source actor
- ONE action
- ONE target actor
- Time information (if available)
- Location information (if available)

Rules:
1. Each atomic fact must be a single, self-contained sentence.
2. Do NOT merge multiple events into one fact.
3. Preserve all information exactly as it appears.
4. Return ONLY a JSON array of strings.

Text: {text}

Output:"""

        try:
            content = self._call_llm(prompt)
            start = content.find('[')
            end = content.rfind(']') + 1
            if start == -1 or end == 0:
                return [text]
            result = json.loads(content[start:end])
            return result if isinstance(result, list) else [text]
        except:
            return [text]

    def extract_entities(self, fact):
        """Step 2: Entity Extraction (CoT + Domain Balance)"""
        prompt = f"""From this sentence, identify the TWO main actors involved.

Sentence: "{fact}"

Think step by step:
1. Who is PERFORMING the action? (entity1)
2. Who is ULTIMATELY AFFECTED by the action? (entity2)

CRITICAL RULES for entity2:
- entity2 must be a PERSON, COUNTRY, ORGANIZATION, or COMPANY
- NOT a concept (unemployment, sovereignty, globalization, literacy)
- NOT a geographic feature (Sahara Desert, Baltic Sea, Kamchatka Peninsula)
- NOT an object (rockets, tariffs, treaty, loan, smartphone)
- NOT an event name (Davos Forum, APEC summit, Clean Air Act)
- If the action targets a CITY, entity2 = the COUNTRY it belongs to
- If the action targets a SUB-ORGANIZATION, entity2 = the parent COUNTRY or ORG
- If a government acts on its own people/economy, entity2 = that country itself
- If two entities act together, entity2 = the PARTNER
- If a natural disaster strikes a place, entity2 = the COUNTRY affected

=== POLITICS/MILITARY examples ===

"Pakistan carried out airstrikes on Khost"
→ Khost is a city in Afghanistan
→ {{"entity1": "Pakistan", "entity2": "Afghanistan"}}

"United Kingdom sanctioned Myanmar's Tatmadaw"
→ The Tatmadaw is part of Myanmar
→ {{"entity1": "United Kingdom", "entity2": "Myanmar"}}

"Morocco condemned Sanchez's statement on Western Sahara"
→ Sanchez is a Spanish official. Morocco condemned the Spanish position.
→ {{"entity1": "Morocco", "entity2": "Spain"}}

"Argentina applied to join BRICS"
→ {{"entity1": "Argentina", "entity2": "BRICS"}}

"Japan and Australia held joint naval exercises"
→ {{"entity1": "Japan", "entity2": "Australia"}}

=== BUSINESS/TECHNOLOGY examples ===

"Salesforce acquired Slack"
→ {{"entity1": "Salesforce", "entity2": "Slack"}}

"Netflix opened a new production studio"
→ {{"entity1": "Netflix", "entity2": "Netflix"}}

"Spotify cut 1,500 jobs"
→ {{"entity1": "Spotify", "entity2": "Employees"}}

"Sony unveiled the PlayStation 5 console"
→ {{"entity1": "Sony", "entity2": "PlayStation 5"}}

"Renault partnered with Qualcomm"
→ {{"entity1": "Renault", "entity2": "Qualcomm"}}

"WeWork ousted Adam Neumann"
→ {{"entity1": "WeWork", "entity2": "Adam Neumann"}}

=== DISASTER/ENVIRONMENT examples ===

"A powerful earthquake hit Petrinja in Croatia"
→ {{"entity1": "Earthquake", "entity2": "Croatia"}}

"Typhoon Haiyan made landfall near Tacloban, Philippines"
→ {{"entity1": "Typhoon Haiyan", "entity2": "Philippines"}}

"Portugal mobilized firefighters against wildfires in Leiria"
→ {{"entity1": "Portugal", "entity2": "Portugal"}}

"World Food Programme airlifted supplies to Somalia"
→ {{"entity1": "World Food Programme", "entity2": "Somalia"}}

"Thailand rescued villagers from floods in Chiang Rai"
→ {{"entity1": "Thailand", "entity2": "Flood victims"}}

Now extract from:
"{fact}"

Return ONLY a JSON object:"""

        try:
            content = self._call_llm(prompt)
            start = content.find('{')
            end = content.rfind('}') + 1
            if start == -1 or end == 0:
                return None
            return json.loads(content[start:end])
        except:
            return None

    def extract_relation(self, fact, entity1, entity2):
        """Step 3: Relation Extraction"""
        prompt = f"""What action did "{entity1}" perform toward "{entity2}" in this sentence?

Sentence: "{fact}"

Rules:
1. Describe the core action in 2-5 words
2. Use base verb form (e.g. "Provide" not "provided")
3. Focus on the action TYPE, not specific details
4. Return ONLY a JSON object

Examples:
"Pakistan carried out airstrikes on Afghanistan" → {{"relation": "Conduct airstrikes"}}
"United Kingdom froze the assets of Belarusian officials" → {{"relation": "Freeze assets"}}
"Salesforce acquired Slack" → {{"relation": "Acquire"}}
"Typhoon Haiyan battered the Philippines" → {{"relation": "Strike"}}
"Portugal mobilized firefighters" → {{"relation": "Mobilize firefighters"}}
"World Food Programme airlifted supplies" → {{"relation": "Deliver supplies"}}
"Netflix opened a production studio" → {{"relation": "Open facility"}}
"Spotify cut 1,500 jobs" → {{"relation": "Cut jobs"}}

entity1: {entity1}
entity2: {entity2}
Output:"""

        try:
            content = self._call_llm(prompt)
            start = content.find('{')
            end = content.rfind('}') + 1
            if start == -1 or end == 0:
                return "unknown"
            result = json.loads(content[start:end])
            return result.get('relation', 'unknown')
        except:
            return "unknown"

    def extract_time_location(self, text):
        """Step 4: Time/Location Extraction (from original text)"""
        prompt = f"""Extract the time and location from this sentence.

Sentence: "{text}"

Rules:
1. time: Extract date in YYYY-MM-DD format. Use "unknown" if not found.
2. location: Extract the country where the event occurred. Use "unknown" if not found.
3. For location, return the COUNTRY name, not city.
4. Return ONLY a JSON object.

Examples:
"... on November 30, 2015 in Lima, Peru" → {{"time": "2015-11-30", "location": "Peru"}}
"... on September 3, 2017 in Vienna" → {{"time": "2017-09-03", "location": "Austria"}}
"... in Chiang Mai, Thailand" → {{"time": "unknown", "location": "Thailand"}}
"... near Petrinja, Croatia" → {{"time": "unknown", "location": "Croatia"}}

Output:"""

        try:
            content = self._call_llm(prompt)
            start = content.find('{')
            end = content.rfind('}') + 1
            if start == -1 or end == 0:
                return {"time": "unknown", "location": "unknown"}
            return json.loads(content[start:end])
        except:
            return {"time": "unknown", "location": "unknown"}

    def extract_with_trace(self, text):
        """
        Multi-stage pipeline that also returns the intermediate atomic facts.
        Returns {"atomic_facts": [...], "quintuples": [...]}.
        """
        atomic_facts = self.decompose_to_atomic_facts(text)
        quintuples = self._extract_from_facts(text, atomic_facts)
        return {"atomic_facts": atomic_facts, "quintuples": quintuples}

    def _extract_single(self, text):
        """Single extraction (internal use)"""
        atomic_facts = self.decompose_to_atomic_facts(text)
        return self._extract_from_facts(text, atomic_facts)

    def _extract_from_facts(self, text, atomic_facts):
        """Steps 2-4 for each atomic fact"""
        quintuples = []
        for fact in atomic_facts:
            if not isinstance(fact, str):
                fact = str(fact)
            entities = self.extract_entities(fact)
            if not entities:
                continue

            entity1 = entities.get('entity1', 'unknown')
            entity2 = entities.get('entity2', 'unknown')

            if not entity1 or entity1 == 'unknown':
                continue

            relation = self.extract_relation(fact, entity1, entity2)
            # Extract Time/Location from original text (prevent information loss)
            time_loc = self.extract_time_location(text)

            quintuple = {
                "entity1": entity1,
                "relation": relation,
                "entity2": entity2,
                "time": time_loc.get('time', 'unknown'),
                "location": time_loc.get('location', 'unknown')
            }
            quintuples.append(quintuple)

        return quintuples

    def _majority_vote(self, candidates, field):
        """Majority voting"""
        values = [str(c.get(field, 'unknown')).lower().strip()
                  for c in candidates if c]

        if not values:
            return 'unknown'

        counter = Counter(values)
        most_common = counter.most_common(1)[0][0]

        for c in candidates:
            if c and str(c.get(field, '')).lower().strip() == most_common:
                return str(c.get(field, ''))

        return most_common

    def extract_quintuples(self, text):
        """Multi-stage pipeline (single extraction)"""
        return self._extract_single(text)

    def extract_quintuples_sc(self, text, n_samples=3):
        """Self-Consistency extraction (n extractions → majority vote)"""
        all_results = []
        for _ in range(n_samples):
            result = self._extract_single(text)
            if result:
                all_results.append(result[0])
            else:
                all_results.append(None)

        valid = [r for r in all_results if r is not None]
        if not valid:
            return []

        fields = ['entity1', 'relation', 'entity2', 'time', 'location']
        final = {}
        for field in fields:
            final[field] = self._majority_vote(valid, field)

        return [final]

    def extract_direct(self, text):
        """Legacy method (direct extraction in one step) - for comparison experiments"""
        prompt = f"""Extract spatio-temporal quintuples from the text.

A quintuple has 5 elements:
- entity1: source actor
- relation: action
- entity2: target actor
- time: YYYY-MM-DD
- location: country

Return ONLY a JSON array.

Text: {text}
Output:"""

        try:
            content = self._call_llm(prompt)
            start = content.find('[')
            end = content.rfind(']') + 1
            if start == -1 or end == 0:
                return []
            result = json.loads(content[start:end])
            return result if isinstance(result, list) else []
        except:
            return []

    def extract_from_icews_row(self, row):
        """Extract quintuple from ICEWS data row (preserve original)"""
        try:
            city = str(row.get('City', '')).strip()
            province = str(row.get('Province', '')).strip()
            country = str(row.get('Country', '')).strip()

            location_parts = [
                p for p in [city, province, country]
                if p and p != 'nan'
            ]
            location_str = ', '.join(location_parts) \
                if location_parts else 'unknown'

            text = (
                f"{row['Source Name']} {row['Event Text']} "
                f"{row['Target Name']} "
                f"on {row['Event Date']} in {location_str}."
            )

            original_entity1 = str(row.get('Source Name', ''))
            original_relation = str(row.get('Event Text', ''))
            original_entity2 = str(row.get('Target Name', ''))
            original_time = str(row.get('Event Date', ''))

            time_loc = self.extract_time_location(text)

            quintuple = {
                'entity1': original_entity1,
                'relation': original_relation,
                'entity2': original_entity2,
                'time': original_time,
                'location': time_loc.get('location', location_str)
            }

            return [quintuple]

        except Exception as e:
            print(f"Error processing row: {e}")
            return []
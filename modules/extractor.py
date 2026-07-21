import ollama
import json
from collections import Counter
from config import LLM_MODEL, LLM_HOST

class STKGExtractor:
    """
    Module 1: Knowledge Extraction
    Multi-stage Extraction + CoT + Self-Consistency
    Time/Location extracted from original text (prevent information loss)
    """

    def __init__(self, model=LLM_MODEL, host=LLM_HOST):
        self.model = model
        self.client = ollama.Client(host=host)
        self.prompt_version = "v11_timeloc_fix"

    def _call_llm(self, prompt):
        """Common function for LLM API calls"""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        except Exception as e:
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
- NOT a concept (inflation, democracy, climate, deforestation)
- NOT a geographic feature (Amazon, East Sea, Noto Peninsula)
- NOT an object (missiles, sanctions, agreement, package, headset)
- NOT an event name (COP27, G20, CHIPS Act)
- If the action targets a CITY, entity2 = the COUNTRY it belongs to
- If the action targets a SUB-ORGANIZATION, entity2 = the parent COUNTRY or ORG
- If a government acts on its own people/economy, entity2 = that country itself
- If two entities act together, entity2 = the PARTNER
- If a natural disaster strikes a place, entity2 = the COUNTRY affected

=== POLITICS/MILITARY examples ===

"Russia launched missiles at Mykolaiv"
→ Mykolaiv is a city in Ukraine
→ {{"entity1": "Russia", "entity2": "Ukraine"}}

"EU sanctioned Iran's IRGC"
→ IRGC is part of Iran
→ {{"entity1": "European Union", "entity2": "Iran"}}

"China condemned Pelosi's visit to Taiwan"
→ Pelosi is a US official. China condemned the US action.
→ {{"entity1": "China", "entity2": "United States"}}

"Finland applied to join NATO"
→ {{"entity1": "Finland", "entity2": "NATO"}}

"South Korea and US conducted joint air drills"
→ {{"entity1": "South Korea", "entity2": "United States"}}

=== BUSINESS/TECHNOLOGY examples ===

"Microsoft acquired Activision Blizzard"
→ {{"entity1": "Microsoft", "entity2": "Activision Blizzard"}}

"Tesla announced construction of a new Gigafactory"
→ {{"entity1": "Tesla", "entity2": "Tesla"}}

"Amazon laid off 18,000 employees"
→ {{"entity1": "Amazon", "entity2": "Employees"}}

"Apple launched the Vision Pro headset"
→ {{"entity1": "Apple", "entity2": "Vision Pro"}}

"Toyota partnered with Nvidia"
→ {{"entity1": "Toyota", "entity2": "Nvidia"}}

"OpenAI fired Sam Altman"
→ {{"entity1": "OpenAI", "entity2": "Sam Altman"}}

=== DISASTER/ENVIRONMENT examples ===

"A magnitude 7.8 earthquake struck southern Turkey"
→ {{"entity1": "Earthquake", "entity2": "Turkey"}}

"Hurricane Otis made landfall near Acapulco, Mexico"
→ {{"entity1": "Hurricane Otis", "entity2": "Mexico"}}

"Canada declared emergency over wildfires in British Columbia"
→ {{"entity1": "Canada", "entity2": "Canada"}}

"United Nations launched emergency aid appeal for Afghanistan"
→ {{"entity1": "United Nations", "entity2": "Afghanistan"}}

"India rescued thousands from floods in Assam"
→ {{"entity1": "India", "entity2": "Flood victims"}}

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
"Russia launched missiles at Ukraine" → {{"relation": "Launch missiles"}}
"EU sanctioned Iran" → {{"relation": "Impose sanctions"}}
"Microsoft acquired Activision" → {{"relation": "Acquire"}}
"Earthquake struck Turkey" → {{"relation": "Strike"}}
"Canada declared emergency" → {{"relation": "Declare emergency"}}
"UN launched aid appeal" → {{"relation": "Launch aid appeal"}}
"Tesla announced construction" → {{"relation": "Announce construction"}}
"Amazon laid off employees" → {{"relation": "Lay off employees"}}

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
"... on October 13, 2022 in Washington D.C." → {{"time": "2022-10-13", "location": "United States"}}
"... on October 27, 2022 in Brussels" → {{"time": "2022-10-27", "location": "Belgium"}}
"... in Kumamoto, Japan" → {{"time": "unknown", "location": "Japan"}}
"... near Acapulco, Mexico" → {{"time": "unknown", "location": "Mexico"}}

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

    def _extract_single(self, text):
        """Single extraction (internal use)"""
        atomic_facts = self.decompose_to_atomic_facts(text)

        quintuples = []
        for fact in atomic_facts:
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
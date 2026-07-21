import ollama
import json
from modules.graph import STKGBuilder
from config import LLM_MODEL, LLM_HOST, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class STKGQueryEngine:
    """
    Module 4: Query and Reasoning
    STKG-based natural language query processing
    - Support for single/complex queries
    - Support for Korean/English queries
    - Spatio-temporal reasoning
    - STMH-LLM reference: Spatio-temporal multi-hop reasoning
    """

    def __init__(self,
                 model=LLM_MODEL,
                 host=LLM_HOST,
                 uri=NEO4J_URI,
                 user=NEO4J_USER,
                 password=NEO4J_PASSWORD):
        self.model = model
        self.client = ollama.Client(host=host)
        self.builder = STKGBuilder(uri, user, password)

    def parse_query(self, user_input):
        """
        Convert natural language query to STKG search parameters
        Support for both Korean/English
        Extract complex conditions
        """
        prompt = f"""You are a query parser for a spatio-temporal knowledge graph.

Analyze the user's question and extract ALL search parameters.

Return ONLY a JSON object:
{{
  "query_type": "entity" or "time" or "location" or "complex" or "stats",
  "entity": "entity name if mentioned, else null",
  "time": "date in YYYY-MM-DD if mentioned, else null",
  "location": "location if mentioned, else null",
  "intent": "what the user wants to know"
}}

Examples:
"What did Israel do on 2022-10-14?" → {{"query_type":"complex","entity":"Israel","time":"2022-10-14","location":null,"intent":"actions of entity on date"}}
"이스라엘이 2022년 10월에 한 행동은?" → {{"query_type":"complex","entity":"Israel","time":"2022-10","location":null,"intent":"actions of entity in time period"}}
"미국에서 일어난 사건은?" → {{"query_type":"location","entity":null,"time":null,"location":"United States","intent":"events in location"}}
"어떤 나라가 가장 많이 언급됐나?" → {{"query_type":"stats","entity":null,"time":null,"location":null,"intent":"most mentioned country"}}

User question: {user_input}
JSON:"""

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response['message']['content']
            start = content.find('{')
            end = content.rfind('}') + 1
            return json.loads(content[start:end])
        except Exception:
            return {
                "query_type": "general",
                "entity": None,
                "time": None,
                "location": None,
                "intent": "unknown"
            }

    def search_stkg(self, parsed):
        """
        Search STKG with parsed parameters
        Support for complex conditions - utilize Neo4j complex queries
        """
        try:
            entity = parsed.get('entity')
            time = parsed.get('time')
            location = parsed.get('location')
            query_type = parsed.get('query_type', 'general')

            # Statistics query
            if query_type == 'stats':
                return self._handle_stats_query(parsed)

            # Complex query (entity + time + location combination)
            if sum(x is not None for x in [entity, time, location]) >= 1:
                return self.builder.query_complex(
                    entity=entity,
                    time=time,
                    location=location
                )

            return []

        except Exception as e:
            print(f"STKG search error: {e}")
            return []

    def _handle_stats_query(self, parsed):
        """Handle statistics query"""
        try:
            with self.builder.driver.session() as session:
                # Most mentioned entity
                result = session.run("""
                    MATCH (e:Entity)-[r:RELATION]->()
                    RETURN e.name as entity, count(r) as count
                    ORDER BY count DESC
                    LIMIT 10
                """)
                return [
                    {
                        "entity1": record["entity"],
                        "relation": "mentioned",
                        "entity2": f"{record['count']} times",
                        "time": "",
                        "location": ""
                    }
                    for record in result
                ]
        except Exception as e:
            print(f"Statistics query error: {e}")
            return []

    def generate_answer(self, user_input, results):
        """
        Generate LLM answer based on STKG search results
        Answer in the same language as user query
        """
        if not results:
            return "No related information found in STKG."

        facts = "\n".join([
            f"- {r['entity1']} {r['relation']} {r['entity2']} "
            f"on {r['time']} in {r['location']}"
            for r in results[:5]
        ])

        prompt = f"""You are an expert analyst of a spatio-temporal knowledge graph.

STRICT RULES:
1. Answer ONLY using the facts provided below.
2. Do NOT add any information not in the facts.
3. Do NOT infer or assume anything not explicitly stated.
4. Answer in the SAME LANGUAGE as the question.
5. Keep it concise and factual.
6. If the question is in Korean, answer in Korean.
7. If the question is in English, answer in English.

Facts from STKG:
{facts}

Question: {user_input}

Answer (based ONLY on the facts above):"""

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        except Exception as e:
            print(f"Answer generation error: {e}")
            return "An error occurred while generating the answer."

    def query(self, user_input):
        """Complete query pipeline"""
        parsed = self.parse_query(user_input)
        results = self.search_stkg(parsed)
        answer = self.generate_answer(user_input, results)
        return answer, results, parsed

    def close(self):
        """Clean up resources"""
        self.builder.close()
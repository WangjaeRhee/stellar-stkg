from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class STKGBuilder:
    """
    Module 3: STKG Construction
    Store quintuples in Neo4j graph database
    - Entity → Node (name, type properties)
    - Relation → Edge (relation, time, location properties)
    - Index optimization
    """

    def __init__(self,
                 uri=NEO4J_URI,
                 user=NEO4J_USER,
                 password=NEO4J_PASSWORD):
        try:
            self.driver = GraphDatabase.driver(
                uri, auth=(user, password)
            )
        except Exception as e:
            raise ConnectionError(f"Neo4j connection failed: {e}")

    def close(self):
        """Close driver"""
        if self.driver:
            self.driver.close()

    def clear_db(self):
        """Clear database"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared")
        except Exception as e:
            print(f"Database clear error: {e}")

    def create_indexes(self):
        """Create indexes for performance improvement"""
        try:
            with self.driver.session() as session:
                session.run(
                    "CREATE INDEX entity_name IF NOT EXISTS "
                    "FOR (e:Entity) ON (e.name)"
                )
                session.run(
                    "CREATE INDEX relation_time IF NOT EXISTS "
                    "FOR ()-[r:RELATION]-() ON (r.time)"
                )
                session.run(
                    "CREATE INDEX relation_location IF NOT EXISTS "
                    "FOR ()-[r:RELATION]-() ON (r.location)"
                )
            print("Indexes created")
        except Exception as e:
            print(f"Index creation error: {e}")

    def insert_quintuple(self, q):
        """
        Insert single quintuple
        Entity node + RELATION edge
        """
        entity1 = q.get('entity1', '')
        entity2 = q.get('entity2', '')

        if not entity1 or not entity2:
            return False

        relation = q.get('relation', 'unknown')
        time = q.get('time', 'unknown')
        location = q.get('location', 'unknown')

        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (e1:Entity {name: $entity1})
                    MERGE (e2:Entity {name: $entity2})
                    CREATE (e1)-[r:RELATION {
                        relation: $relation,
                        time: $time,
                        location: $location
                    }]->(e2)
                """, {
                    "entity1": entity1,
                    "entity2": entity2,
                    "relation": relation,
                    "time": time,
                    "location": location
                })
            return True
        except Exception as e:
            print(f"Insert failed: {e}")
            return False

    def insert_batch(self, quintuples):
        """Batch insert"""
        success = 0
        fail = 0
        for q in quintuples:
            if self.insert_quintuple(q):
                success += 1
            else:
                fail += 1
        return success, fail

    def get_stats(self):
        """Return STKG statistics"""
        try:
            with self.driver.session() as session:
                node_count = session.run(
                    "MATCH (n) RETURN count(n) as count"
                ).single()['count']
                edge_count = session.run(
                    "MATCH ()-[r]->() RETURN count(r) as count"
                ).single()['count']
                unique_locations = session.run(
                    "MATCH ()-[r:RELATION]->() "
                    "RETURN count(DISTINCT r.location) as count"
                ).single()['count']
                unique_times = session.run(
                    "MATCH ()-[r:RELATION]->() "
                    "RETURN count(DISTINCT r.time) as count"
                ).single()['count']
            return {
                "nodes": node_count,
                "edges": edge_count,
                "unique_locations": unique_locations,
                "unique_times": unique_times
            }
        except Exception as e:
            print(f"Statistics query error: {e}")
            return {"nodes": 0, "edges": 0,
                    "unique_locations": 0, "unique_times": 0}

    def query_by_entity(self, entity_name):
        """Entity-based query"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e1:Entity)-[r:RELATION]->(e2:Entity)
                    WHERE toLower(e1.name) CONTAINS toLower($name)
                       OR toLower(e2.name) CONTAINS toLower($name)
                    RETURN e1.name as entity1, r.relation as relation,
                           e2.name as entity2, r.time as time,
                           r.location as location
                    ORDER BY r.time
                """, {"name": entity_name})
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Query error: {e}")
            return []

    def query_by_time(self, time):
        """Time-based query"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e1:Entity)-[r:RELATION]->(e2:Entity)
                    WHERE r.time CONTAINS $time
                    RETURN e1.name as entity1, r.relation as relation,
                           e2.name as entity2, r.time as time,
                           r.location as location
                    ORDER BY r.time
                """, {"time": time})
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Query error: {e}")
            return []

    def query_by_location(self, location):
        """Location-based query"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e1:Entity)-[r:RELATION]->(e2:Entity)
                    WHERE toLower(r.location) CONTAINS toLower($location)
                    RETURN e1.name as entity1, r.relation as relation,
                           e2.name as entity2, r.time as time,
                           r.location as location
                    ORDER BY r.time
                """, {"location": location})
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Query error: {e}")
            return []

    def query_complex(self, entity=None, time=None, location=None):
        """
        Complex query
        Combined conditions: entity + time + location
        """
        conditions = []
        params = {}

        if entity:
            conditions.append(
                "(toLower(e1.name) CONTAINS toLower($entity) "
                "OR toLower(e2.name) CONTAINS toLower($entity))"
            )
            params["entity"] = entity

        if time:
            conditions.append("r.time CONTAINS $time")
            params["time"] = time

        if location:
            conditions.append(
                "toLower(r.location) CONTAINS toLower($location)"
            )
            params["location"] = location

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        try:
            with self.driver.session() as session:
                result = session.run(f"""
                    MATCH (e1:Entity)-[r:RELATION]->(e2:Entity)
                    WHERE {where_clause}
                    RETURN e1.name as entity1, r.relation as relation,
                           e2.name as entity2, r.time as time,
                           r.location as location
                    ORDER BY r.time
                """, params)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Complex query error: {e}")
            return []
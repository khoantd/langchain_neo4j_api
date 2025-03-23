from typing import Optional, List
from langchain_core.documents import Document
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from common.config import Config

class Neo4jVectorManager:
    def __init__(self, 
                 uri: str = Config.NEO4J_URI,
                 username: str = Config.NEO4J_USERNAME,
                 password: str = Config.NEO4J_PASSWORD):
        self.uri = uri
        self.username = username
        self.password = password
        self.embeddings = OpenAIEmbeddings()
        # Get the embedding dimensions from the embedding function
        self.embedding_dimensions = self.embeddings.dimensions
        
    def insert_text(self, 
                   text: str, 
                   existing_index: Optional[str] = None, 
                   chunk_size: int = 1000, 
                   chunk_overlap: int = 0) -> Neo4jVector:
        """
        Insert text into Neo4j vector database using OpenAI embeddings
        
        Args:
            text (str): The text content to be inserted
            existing_index (str, optional): Name of existing index to use
            chunk_size (int): Size of text chunks (default: 1000)
            chunk_overlap (int): Overlap between chunks (default: 0)
        """
        # Create a Document object from the text
        document = Document(page_content=text)
        
        # Split the text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents([document])
        
        return self._store_documents(docs, existing_index)
    
    def _store_documents(self, 
                        docs: List[Document], 
                        existing_index: Optional[str] = None) -> Neo4jVector:
        """
        Store documents in Neo4j with vector embeddings
        
        Args:
            docs (List[Document]): Documents to store
            existing_index (str, optional): Name of existing index to use
        """
        if existing_index:
            return Neo4jVector.from_existing_index(
                docs,
                self.embeddings,
                url=self.uri,
                username=self.username,
                password=self.password,
                index_name=existing_index)
        
        return Neo4jVector.from_documents(
            docs,
            self.embeddings,
            url=self.uri,
            username=self.username,
            password=self.password)
    
    def load_existing_index(self, index_name: str) -> Neo4jVector:
        """
        Load an existing Neo4j vector index
        
        Args:
            index_name (str): Name of existing index to use
        """
        from neo4j import GraphDatabase
        
        # First check the index dimensions
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        try:
            with driver.session() as session:
                # Get the index configuration
                config_query = """
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties, options
                WHERE name = $index_name AND type = 'VECTOR'
                RETURN options
                """
                result = session.run(config_query, index_name=index_name)
                config = result.single()
                
                if not config:
                    raise ValueError(f"Vector index '{index_name}' not found")
                
                index_dimensions = config["options"]["indexConfig"]["vector.dimensions"]
                
                # For OpenAI embeddings, we can't change dimensions
                # Instead, we'll use the default OpenAI dimensions (1536)
                if index_dimensions != self.embedding_dimensions:
                    print(f"Warning: Index '{index_name}' has dimensions {index_dimensions} "
                          f"but OpenAI embeddings have fixed dimensions {self.embedding_dimensions}. "
                          "This might cause issues with vector operations.")
        
        finally:
            driver.close()
        
        return Neo4jVector.from_existing_index(
            self.embeddings,
            url=self.uri,
            username=self.username,
            password=self.password,
            index_name=index_name)
    
    def load_from_graph(self,
                       index_name: str,
                       node_label: str,
                       text_node_properties: List[str],
                       embedding_node_property: str) -> Neo4jVector:
        """
        Initialize from an existing Neo4j graph with vector index
        
        Args:
            index_name (str): Name of existing vector index
            node_label (str): Label of nodes containing text and embeddings
            text_node_properties (List[str]): Node properties containing text to search
            embedding_node_property (str): Node property containing embeddings
        """
        from neo4j import GraphDatabase
        
        # First check for problematic nodes
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        try:
            with driver.session() as session:
                # Build the text property by concatenating the specified properties
                text_properties = [f"n.{prop}" for prop in text_node_properties]
                text_concat = " + ' ' + ".join(f"COALESCE(n.{prop}, '')" for prop in text_node_properties)
                
                # First, ensure all nodes have a text property
                ensure_text_query = f"""
                MATCH (n:{node_label})
                WHERE n.text IS NULL OR n.text = ''
                SET n.text = {text_concat}
                """
                session.run(ensure_text_query)
                
                # Then proceed with loading the graph
                return Neo4jVector.from_existing_graph(
                    self.embeddings,
                    url=self.uri,
                    username=self.username,
                    password=self.password,
                    index_name=index_name,
                    node_label=node_label,
                    text_node_properties=["text"],  # Use the text property we just ensured exists
                    embedding_node_property=embedding_node_property)
                
        finally:
            driver.close()

    def check_text_properties(self, node_label: str, text_properties: List[str]) -> List[dict]:
        """
        Check for nodes with missing or empty text properties
        
        Args:
            node_label (str): Label of nodes to check
            text_properties (List[str]): List of text properties to verify
            
        Returns:
            List[dict]: List of nodes with issues
        """
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        try:
            with driver.session() as session:
                # First check for the 'text' property specifically
                text_query = f"""
                MATCH (n:{node_label})
                WHERE n.text IS NULL OR n.text = ''
                RETURN n
                """
                
                result = session.run(text_query)
                problematic_nodes = []
                
                for record in result:
                    node = record["n"]
                    node_data = {
                        "id": node.id,
                        "labels": list(node.labels),
                        "properties": dict(node),
                        "missing_property": "text"
                    }
                    problematic_nodes.append(node_data)
                
                # Then check for other specified properties
                for prop in text_properties:
                    if prop != "text":  # Skip 'text' as we already checked it
                        property_query = f"""
                        MATCH (n:{node_label})
                        WHERE n.{prop} IS NULL OR n.{prop} = ''
                        RETURN n
                        """
                        
                        result = session.run(property_query)
                        for record in result:
                            node = record["n"]
                            node_data = {
                                "id": node.id,
                                "labels": list(node.labels),
                                "properties": dict(node),
                                "missing_property": prop
                            }
                            problematic_nodes.append(node_data)
                
                return problematic_nodes
                
        finally:
            driver.close()

    def list_indexes(self) -> List[dict]:
        """
        List all vector indexes in Neo4j
        
        Returns:
            List[dict]: List of index information including name, configuration, and dimensions
        """
        from neo4j import GraphDatabase
        from typing import List, Dict
        
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        try:
            with driver.session() as session:
                # Query to get all vector indexes with their configurations
                result = session.run("""
                    SHOW INDEXES
                    YIELD name, type, labelsOrTypes, properties, options
                    WHERE type = 'VECTOR'
                    RETURN name, labelsOrTypes, properties, options
                """)
                
                indexes = []
                for record in result:
                    # Extract dimensions from the index configuration
                    dimensions = None
                    if record["options"] and "indexConfig" in record["options"]:
                        dimensions = record["options"]["indexConfig"].get("vector.dimensions")
                    
                    index_info = {
                        "name": record["name"],
                        "labels_or_types": record["labelsOrTypes"],
                        "properties": record["properties"],
                        "type": "VECTOR",
                        "dimensions": dimensions,
                        "embedding_model_dimensions": self.embedding_dimensions,
                        "dimensions_match": dimensions == self.embedding_dimensions if dimensions is not None else False
                    }
                    indexes.append(index_info)
                
                return indexes
                
        finally:
            driver.close()

    def fix_text_properties(self, node_label: str, text_properties: List[str]) -> dict:
        """
        Automatically fix nodes with missing or empty text properties by concatenating specified properties
        
        Args:
            node_label (str): Label of nodes to fix
            text_properties (List[str]): List of text properties to concatenate
            
        Returns:
            dict: Summary of fixes applied
        """
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        try:
            with driver.session() as session:
                # First, find all problematic nodes
                problematic_query = f"""
                MATCH (n:{node_label})
                WHERE {' OR '.join(f'n.{prop} IS NULL OR n.{prop} = ""' for prop in text_properties)}
                RETURN n
                """
                
                result = session.run(problematic_query)
                problematic_nodes = []
                for record in result:
                    node = record["n"]
                    problematic_nodes.append({
                        "id": node.id,
                        "labels": list(node.labels),
                        "properties": dict(node)
                    })
                
                if not problematic_nodes:
                    return {
                        "status": "success",
                        "message": "No problematic nodes found",
                        "fixed_count": 0
                    }
                
                # Build the concatenation expression with COALESCE for each property
                concat_expr = " + ' ' + ".join(f"COALESCE(n.{prop}, '')" for prop in text_properties)
                
                # Fix the nodes by setting the text property
                fix_query = f"""
                MATCH (n:{node_label})
                WHERE {' OR '.join(f'n.{prop} IS NULL OR n.{prop} = ""' for prop in text_properties)}
                SET n.text = {concat_expr}
                RETURN n
                """
                
                result = session.run(fix_query)
                fixed_nodes = []
                for record in result:
                    node = record["n"]
                    fixed_nodes.append({
                        "id": node.id,
                        "labels": list(node.labels),
                        "properties": dict(node)
                    })
                
                return {
                    "status": "success",
                    "message": f"Fixed {len(fixed_nodes)} nodes",
                    "fixed_count": len(fixed_nodes),
                    "fixed_nodes": fixed_nodes
                }
                
        finally:
            driver.close()

    def create_index(self, 
                    index_name: str,
                    node_label: str,
                    embedding_property: str,
                    dimensions: Optional[int] = None) -> dict:
        """
        Create a new vector index in Neo4j
        
        Args:
            index_name (str): Name of the index to create
            node_label (str): Label of nodes to index
            embedding_property (str): Property containing the embeddings
            dimensions (int, optional): Dimension of the embeddings. If not provided, uses the embedding function's dimensions
            
        Returns:
            dict: Information about the created index
        """
        from neo4j import GraphDatabase
        
        # Use the embedding function's dimensions if not specified
        if dimensions is None:
            dimensions = self.embedding_dimensions
        
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        try:
            with driver.session() as session:
                # Check if index already exists
                check_query = """
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties
                WHERE name = $index_name
                RETURN name, type, labelsOrTypes, properties
                """
                result = session.run(check_query, index_name=index_name)
                existing_index = result.single()
                
                if existing_index:
                    # If index exists, verify its dimensions
                    if existing_index["type"] == "VECTOR":
                        # Get the current index configuration
                        config_query = """
                        SHOW INDEXES
                        YIELD name, type, labelsOrTypes, properties, options
                        WHERE name = $index_name
                        RETURN options
                        """
                        config_result = session.run(config_query, index_name=index_name)
                        config = config_result.single()["options"]
                        
                        if config and "indexConfig" in config:
                            current_dimensions = config["indexConfig"].get("vector.dimensions")
                            if current_dimensions != dimensions:
                                return {
                                    "status": "error",
                                    "message": f"Index '{index_name}' exists with different dimensions ({current_dimensions}). Please drop the existing index first."
                                }
                    
                    return {
                        "status": "error",
                        "message": f"Index '{index_name}' already exists"
                    }
                
                # Create the vector index
                create_query = f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:{node_label})
                ON (n.{embedding_property})
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {dimensions},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """
                
                session.run(create_query)
                
                # Verify the index was created
                verify_query = """
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties, options
                WHERE name = $index_name
                RETURN name, type, labelsOrTypes, properties, options
                """
                result = session.run(verify_query, index_name=index_name)
                index_info = result.single()
                
                if index_info:
                    return {
                        "status": "success",
                        "message": f"Index '{index_name}' created successfully",
                        "index": {
                            "name": index_info["name"],
                            "type": index_info["type"],
                            "labels": index_info["labelsOrTypes"],
                            "properties": index_info["properties"],
                            "dimensions": dimensions
                        }
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Failed to create index"
                    }
                
        finally:
            driver.close()

    def search_all_indexes(self, query: str, k: int = 3) -> dict:
        """
        Search for similar text across all existing vector indexes
        
        Args:
            query (str): The search query
            k (int): Number of similar results to return per index
            
        Returns:
            dict: Search results from all indexes
        """
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        try:
            with driver.session() as session:
                # Get all vector indexes
                indexes_query = """
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties, options
                WHERE type = 'VECTOR'
                RETURN name, labelsOrTypes, properties, options
                """
                
                result = session.run(indexes_query)
                indexes = [record for record in result]
                
                if not indexes:
                    return {
                        "status": "error",
                        "message": "No vector indexes found in the database"
                    }
                
                # Search across each index
                all_results = {}
                print("indexes",indexes)
                for index in indexes:
                    index_name = index["name"]
                    try:
                        # Get the index dimensions
                        index_dimensions = index["options"]["indexConfig"]["vector.dimensions"]
                        print("index_dimensions", index_dimensions)
                        
                        # Skip indexes with mismatched dimensions
                        # if index_dimensions != self.embedding_dimensions:
                        #     all_results[index_name] = {
                        #         "error": f"Index dimensions ({index_dimensions}) do not match current embedding model dimensions ({self.embedding_dimensions})"
                        #     }
                        #     continue

                        # self.dimensions=OpenAIEmbeddings(dimensions=index_dimensions)
                        
                        # Load the index and perform search
                        db = self.load_existing_index(index_name)
                        results = db.similarity_search(query, k=k)
                        
                        # Format results
                        formatted_results = [
                            {
                                "content": doc.page_content,
                                "metadata": doc.metadata
                            } for doc in results
                        ]
                        
                        all_results[index_name] = {
                            "node_label": index["labelsOrTypes"][0] if index["labelsOrTypes"] else "Unknown",
                            "results": formatted_results
                        }
                    except Exception as e:
                        all_results[index_name] = {
                            "error": str(e)
                        }
                
                return {
                    "status": "success",
                    "query": query,
                    "results_by_index": all_results
                }
                
        finally:
            driver.close()

    def drop_index(self, index_name: str) -> dict:
        """
        Drop an existing vector index
        
        Args:
            index_name (str): Name of the index to drop
            
        Returns:
            dict: Status of the operation
        """
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        try:
            with driver.session() as session:
                # Check if index exists
                check_query = """
                SHOW INDEXES
                YIELD name, type
                WHERE name = $index_name AND type = 'VECTOR'
                RETURN count(*) as count
                """
                result = session.run(check_query, index_name=index_name)
                if result.single()["count"] == 0:
                    return {
                        "status": "error",
                        "message": f"Vector index '{index_name}' does not exist"
                    }
                
                # Drop the index
                drop_query = f"DROP INDEX {index_name}"
                session.run(drop_query)
                
                return {
                    "status": "success",
                    "message": f"Index '{index_name}' dropped successfully"
                }
                
        finally:
            driver.close()


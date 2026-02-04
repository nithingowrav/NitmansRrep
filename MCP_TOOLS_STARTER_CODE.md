# Starter Code for Databricks & Neo4j MCP Tools

This document contains the complete starter code to run your MCP server.

## 1. Environment Setup

Create a `requirements.txt` file with the following content:

```text
mcp
databricks-vectorsearch
neo4j
python-dotenv
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 2. Configuration

Create a `.env` file in the same directory to store your credentials securely:

```env
# Databricks Configuration
DATABRICKS_HOST=https://<your-workspace-url>
DATABRICKS_TOKEN=dapi...
DATABRICKS_VECTOR_ENDPOINT_NAME=vector-search-endpoint
DATABRICKS_INDEX_NAME=catalog.schema.index_name

# Neo4j Configuration
NEO4J_URI=neo4j+s://<your-instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>
```

## 3. Server Implementation (`server.py`)

Save the following code as `server.py`.

```python
import os
import json
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

# MCP SDK
from mcp.server.fastmcp import FastMCP

# Databricks SDK
from databricks.vector_search.client import VectorSearchClient

# Neo4j Driver
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Initialize the MCP Server
mcp = FastMCP("DatabricksNeo4jTools")

# -----------------------------------------------------------------------------
# GLOBAL CLIENTS INITIALIZATION
# -----------------------------------------------------------------------------
# We initialize clients lazily or globally depending on preference.
# Here we setup connection parameters.

# Databricks Setup
DB_HOST = os.getenv("DATABRICKS_HOST")
DB_TOKEN = os.getenv("DATABRICKS_TOKEN")
DB_ENDPOINT = os.getenv("DATABRICKS_VECTOR_ENDPOINT_NAME")
DB_INDEX = os.getenv("DATABRICKS_INDEX_NAME")

# Neo4j Setup
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# -----------------------------------------------------------------------------
# TOOL 1: DATABRICKS VECTOR SEARCH
# -----------------------------------------------------------------------------

@mcp.tool()
def query_vector_index(
    query_text: str,
    num_results: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> str:
    """
    Queries the Databricks Vector Search index for semantically similar content.

    Args:
        query_text: The question or text to search for.
        num_results: Number of results to return (default: 5).
        filters: Optional dictionary of filters (e.g., {"category": "finance"}).

    Returns:
        A JSON string containing the list of search results with metadata.
    """
    try:
        if not all([DB_HOST, DB_TOKEN, DB_ENDPOINT, DB_INDEX]):
            return "Error: Databricks environment variables are missing."

        # Initialize Client
        client = VectorSearchClient(
            workspace_url=DB_HOST,
            personal_access_token=DB_TOKEN
        )

        # Get the index
        index = client.get_index(endpoint_name=DB_ENDPOINT, index_name=DB_INDEX)

        # Execute Search
        # Note: API signature may vary slightly by version, this is standard.
        results = index.similarity_search(
            query_text=query_text,
            columns=["id", "text_content", "source"], # Adjust based on your schema
            num_results=num_results,
            filters=filters
        )

        # Parse results to a friendly format
        # The SDK usually returns a dictionary. We'll dump it to JSON.
        return json.dumps(results, default=str, indent=2)

    except Exception as e:
        return f"Error querying Databricks Vector Search: {str(e)}"


# -----------------------------------------------------------------------------
# TOOL 2: NEO4J GRAPH TOOLS
# -----------------------------------------------------------------------------

def get_neo4j_driver():
    """Helper to get a Neo4j driver instance."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

@mcp.tool()
def get_graph_schema() -> str:
    """
    Retrieves the schema of the Neo4j graph database to help understand
    available Nodes, Relationships, and Properties.

    Returns:
        A text summary of the graph schema.
    """
    try:
        driver = get_neo4j_driver()

        # Query to get Node Labels and their properties
        node_query = """
        CALL db.schema.nodeTypeProperties()
        YIELD nodeType, propertyName, propertyTypes
        RETURN nodeType, collect(propertyName) as properties
        """

        # Query to get Relationships
        rel_query = """
        CALL db.schema.relTypeProperties()
        YIELD relType, propertyName, propertyTypes
        RETURN relType, collect(propertyName) as properties
        """

        # Query to get connections (Source)-[REL]->(Target)
        # This is expensive on huge graphs, use with caution or limit.
        # A lighter version is db.schema.visualization() but that returns graph objects.
        # We will stick to listing types for safety.

        with driver.session() as session:
            nodes = session.run(node_query).data()
            rels = session.run(rel_query).data()

        driver.close()

        summary = "Graph Schema:\n"
        summary += "Node Types:\n"
        for n in nodes:
            summary += f"- {n['nodeType']}: {n['properties']}\n"

        summary += "\nRelationship Types:\n"
        for r in rels:
            summary += f"- {r['relType']}: {r['properties']}\n"

        return summary

    except Exception as e:
        return f"Error fetching Neo4j schema: {str(e)}"


@mcp.tool()
def execute_cypher_query(query: str) -> str:
    """
    Executes a read-only Cypher query against the Neo4j database.

    Args:
        query: The Cypher query string (e.g., "MATCH (n:Person) RETURN n LIMIT 5").

    Returns:
        The query results as a JSON string.
    """
    # Security Check: Prevent write operations
    forbidden_keywords = ["CREATE", "DELETE", "MERGE", "SET", "REMOVE", "DROP", "DETACH"]
    if any(keyword in query.upper() for keyword in forbidden_keywords):
        return "Error: This tool only allows READ operations. Write keywords are forbidden."

    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run(query)
            data = result.data() # Returns list of dictionaries

        driver.close()
        return json.dumps(data, default=str, indent=2)

    except Exception as e:
        return f"Error executing Cypher query: {str(e)}"

# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # fastmcp handles the server startup and transport
    mcp.run()
```

## 4. How to Run

1.  **Configure**: Ensure your `.env` file is populated.
2.  **Run**:
    ```bash
    python server.py
    ```
3.  **Connect**:
    *   If using an MCP Client (like Claude Desktop or an IDE), configure it to run this script.
    *   Example configuration for Claude Desktop:
        ```json
        {
          "mcpServers": {
            "databricks-neo4j": {
              "command": "python",
              "args": ["/absolute/path/to/server.py"]
            }
          }
        }
        ```

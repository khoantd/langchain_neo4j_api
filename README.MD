# Neo4j Vector API

A FastAPI-based REST API for managing vector operations with Neo4j, supporting embeddings and similarity search capabilities.

## Features

- Vector Index Management
  - Create vector indexes with customizable dimensions
  - List existing vector indexes
  - Drop vector indexes
  - Load existing indexes

- Text Operations
  - Insert text with automatic chunking
  - Load text from graph nodes
  - Search similar text across all indexes
  - Fix and validate text properties

- Authentication
  - JWT-based authentication
  - Secure endpoint access

## API Endpoints

### Authentication
- `POST /api/auth/token` - Get JWT token for authentication

### Vector Operations
- `POST /api/vector/create-index` - Create a new vector index
- `DELETE /api/vector/drop-index/{index_name}` - Drop an existing index
- `GET /api/vector/list-indexes` - List all vector indexes
- `POST /api/vector/insert-text` - Insert text into vector index
- `GET /api/vector/load-index/{index_name}` - Load an existing index
- `POST /api/vector/load-from-graph` - Load vectors from graph nodes
- `POST /api/vector/search` - Search similar text in specific index
- `POST /api/vector/search-all` - Search similar text across all indexes
- `POST /api/vector/check-text-properties` - Check for problematic text properties
- `POST /api/vector/fix-text-properties` - Fix problematic text properties

## Installation

1. Clone the repository
```bash
git clone <repository-url>
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

Create a `config.py` file with the following settings:

```python
class Config:
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "your-password"
    JWT_SECRET_KEY = "your-secret-key"
```

## Usage

### Starting the Server

```bash
uvicorn main:app --reload
```

### Example API Calls

1. Create a Vector Index:
```bash
curl -X POST "http://localhost:8000/api/vector/create-index" \
     -H "Authorization: Bearer your-token" \
     -H "Content-Type: application/json" \
     -d '{
         "index_name": "test_index",
         "node_label": "Document",
         "embedding_property": "embedding"
     }'
```

2. Search Similar Text:
```bash
curl -X POST "http://localhost:8000/api/vector/search" \
     -H "Authorization: Bearer your-token" \
     -H "Content-Type: application/json" \
     -d '{
         "query": "your search query",
         "index_name": "test_index",
         "k": 3
     }'
```

## Dependencies

- FastAPI
- Neo4j
- LangChain
- PyJWT
- OpenAI (for embeddings)

## Notes

- The API uses OpenAI embeddings with 1536 dimensions by default
- Ensure your Neo4j instance supports vector indexes
- JWT tokens are required for all endpoints except authentication

## Error Handling

The API includes comprehensive error handling for:
- Authentication failures
- Invalid index operations
- Dimension mismatches
- Missing or invalid properties
- Database connection issues

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Your License Here]
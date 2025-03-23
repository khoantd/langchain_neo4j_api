from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta

from common.neo4jvector import Neo4jVectorManager
from common.config import Config

router = APIRouter()
security = HTTPBearer()

# Request/Response models
class InsertTextRequest(BaseModel):
    text: str
    existing_index: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 0

class LoadFromGraphRequest(BaseModel):
    index_name: str
    node_label: str
    text_node_properties: List[str]
    embedding_node_property: str

class SearchTextRequest(BaseModel):
    query: str
    index_name: str
    k: int = 3  # Number of similar results to return

class SearchAllIndexesRequest(BaseModel):
    query: str
    k: int = 3  # Number of similar results to return per index

class CheckTextPropertiesRequest(BaseModel):
    node_label: str
    text_properties: List[str]

class CreateIndexRequest(BaseModel):
    index_name: str
    node_label: str
    embedding_property: str
    dimensions: Optional[int] = None  # Optional dimensions, will use embedding function's dimensions if not provided

# JWT Authentication helper
def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.post("/vector/insert-text")
async def insert_text(
    request: InsertTextRequest,
    _: dict = Depends(verify_jwt_token)
):
    try:
        vector_manager = Neo4jVectorManager()
        result = vector_manager.insert_text(
            text=request.text,
            existing_index=request.existing_index,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        return {"status": "success", "message": "Text inserted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vector/load-index/{index_name}")
async def load_existing_index(
    index_name: str,
    _: dict = Depends(verify_jwt_token)
):
    try:
        vector_manager = Neo4jVectorManager()
        result = vector_manager.load_existing_index(index_name)
        return {"status": "success", "message": f"Index '{index_name}' loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vector/load-from-graph")
async def load_from_graph(
    request: LoadFromGraphRequest,
    _: dict = Depends(verify_jwt_token)
):
    try:
        vector_manager = Neo4jVectorManager()
        result = vector_manager.load_from_graph(
            index_name=request.index_name,
            node_label=request.node_label,
            text_node_properties=request.text_node_properties,
            embedding_node_property=request.embedding_node_property
        )
        return {
            "status": "success",
            "message": "Graph loaded successfully",
            "index_name": request.index_name,
            "node_label": request.node_label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vector/search")
async def search_similar_text(
    request: SearchTextRequest,
    _: dict = Depends(verify_jwt_token)
):
    try:
        vector_manager = Neo4jVectorManager()
        # Load the existing index
        db = vector_manager.load_existing_index(request.index_name)
        # Perform similarity search
        results = db.similarity_search(request.query, k=request.k)
        return {
            "status": "success",
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vector/search-all")
async def search_all_indexes(
    request: SearchAllIndexesRequest,
    _: dict = Depends(verify_jwt_token)
):
    try:
        vector_manager = Neo4jVectorManager()
        results = vector_manager.search_all_indexes(
            query=request.query,
            k=request.k
        )
        if results["status"] == "error":
            raise HTTPException(status_code=400, detail=results["message"])
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vector/list-indexes")
async def list_indexes(
    _: dict = Depends(verify_jwt_token)
):
    try:
        vector_manager = Neo4jVectorManager()
        # Get all vector indexes from Neo4j
        indexes = vector_manager.list_indexes()
        return {
            "status": "success",
            "indexes": indexes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vector/check-text-properties")
async def check_text_properties(
    request: CheckTextPropertiesRequest,
    _: dict = Depends(verify_jwt_token)
):
    try:
        vector_manager = Neo4jVectorManager()
        problematic_nodes = vector_manager.check_text_properties(
            node_label=request.node_label,
            text_properties=request.text_properties
        )
        return {
            "status": "success",
            "problematic_nodes": problematic_nodes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vector/fix-text-properties")
async def fix_text_properties(
    request: CheckTextPropertiesRequest,
    _: dict = Depends(verify_jwt_token)
):
    try:
        vector_manager = Neo4jVectorManager()
        result = vector_manager.fix_text_properties(
            node_label=request.node_label,
            text_properties=request.text_properties
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vector/create-index")
async def create_index(
    request: CreateIndexRequest,
    _: dict = Depends(verify_jwt_token)
):
    try:
        vector_manager = Neo4jVectorManager()
        result = vector_manager.create_index(
            index_name=request.index_name,
            node_label=request.node_label,
            embedding_property=request.embedding_property,
            dimensions=request.dimensions
        )
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/vector/drop-index/{index_name}")
async def drop_index(
    index_name: str,
    _: dict = Depends(verify_jwt_token)
):
    try:
        vector_manager = Neo4jVectorManager()
        result = vector_manager.drop_index(index_name)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
    

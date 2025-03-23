import os
from common.config import Config
from common.neo4jvector import *
from fastapi import FastAPI
from api.vector_api import router as vector_router
from api.auth_api import router as auth_router

app = FastAPI()
app.include_router(auth_router, prefix="/api")
app.include_router(vector_router, prefix="/api")

if __name__=="__main__":
    # openai_key=Config.OPENAI_API_KEY
    # print("OpenAI API Key:", Config.NEO4J_URI)
    # Create embedding for System nodes
    vector_manager = Neo4jVectorManager()
    db = vector_manager.load_from_graph('system_index','System',['type','name','description'],'embedding')
    # db = vector_manager.load_existing_index('system_index')
    #  Search for text
    # result = db.similarity_search_with_score("pega Collection", k=1)
    result = db.similarity_search("pega Collection", k=1)
    print(result)



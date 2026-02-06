from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter
import uvicorn
from src.routes.cst_routes import router as cst_router
app = FastAPI()



# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(cst_router, prefix="/cst")



@app.get("/health")
async def health_check():
    return {"status": "ok"}
@app.get("/", tags=["Root"])
async def root():

    return {
        "message" : "Welcome to Turbodiff backend",
        "version" : "1.0.0",
        "endpoints": {
            "docs": "/docs"
        }
    }

# app.include_router(router, prefix="/api/v1")    

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)














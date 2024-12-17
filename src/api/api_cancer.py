from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.abspath(".."))

from models import LungPrediction,llm
app = FastAPI()
 

origins = ['http://localhost:3000']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, 
    allow_methods=['*'],
    allow_headers=['*'], 
)
 
class TestInput(BaseModel):
    message:str

CancerModel = LungPrediction()
@app.post('/')
def root(test_data:TestInput):
    return {"result":test_data}

@app.post("/api/lung")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        embed_vec = CancerModel.get_embedding(file_content)
        result = CancerModel.get_preds_lung(embed_vec)
        response = llm.get_lung_response(file_content,result)
        return {"filename": file.filename, "size": len(file_content),"result":response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
     
@app.post("/api/colon")  
async def upload_file(file: UploadFile = File(...)):
    try: 
        file_content = await file.read() 
        print("done")
        embed_vec = CancerModel.get_embedding(file_content)
        result = CancerModel.get_preds_colon(embed_vec)
        response = llm.get_colon_response(file_content,result)
        return {"filename": file.filename, "size": len(file_content),"result":response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
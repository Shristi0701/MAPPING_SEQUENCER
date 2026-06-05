from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from mapping.similarity import compute_similarity, score_one_pair
from sequencer.topo_sort import build_semester_plan
from mapping.similarity import model, similarity_to_level
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from mapping.evaluator import evaluate_precision_at_k
from fastapi.middleware.cors import CORSMiddleware
from api.pdf_services import generate_accreditation_pdf

app = FastAPI(title="Group 2 CO-PO Mapping API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    id: str
    text: str

class MappingRequest(BaseModel):
    cos: List[Item]
    pos: List[Item]
    psos: List[Item] = []
    peos: List[Item] = []
    top_k: int = 3
    subject: str = ""
    semester: str = ""

@app.get("/")
def root():
    return {"message": "Group 2 CO-PO Mapping API running"}

@app.post("/map/auto")
def map_co_to_po(request: MappingRequest):
    cos = [co.dict() for co in request.cos]
    pos = [po.dict() for po in request.pos]
    results = compute_similarity(cos, pos, top_k=request.top_k)
    return {"mappings": results}


# ----------- SEQUENCER MODELS -----------

class CourseInput(BaseModel):
    id: str
    credits: int = 3
    prerequisites: List[str] = []


class SequencerRequest(BaseModel):
    courses: List[CourseInput]
    max_credits_per_sem: int = 12


# ----------- SEQUENCER ENDPOINT -----------
@app.post("/sequencer/plan")
def generate_plan(request: SequencerRequest):
    courses = [c.dict() for c in request.courses]

    plan, error = build_semester_plan(
        courses,
        max_credits_per_sem=request.max_credits_per_sem
    )

    if error:
        return {"error": error}

    return {
        "total_semesters": len(plan),
        "total_courses": sum(len(s) for s in plan),
        "plan": [
            {
                "semester": i + 1,
                "courses": sem,
                "credits": sum(
                    next(c["credits"] for c in courses if c["id"] == cid)
                    for cid in sem
                )
            }
            for i, sem in enumerate(plan)
        ]
    }


# ----------- MATRIX ENDPOINT -----------

@app.post("/map/matrix")
def mapping_matrix(request: MappingRequest):
    """
    Returns a full CO x (PO + PSO) mapping matrix and a PO x PEO matrix.
    Each cell = mapping level (0, 1, 2, or 3).
    """
    cos = [co.dict() for co in request.cos]
    pos = [po.dict() for po in request.pos]
    psos = [pso.dict() for pso in request.psos]
    peos = [peo.dict() for peo in request.peos]

    # 1. CO-PO Mappings
    co_po_results = compute_similarity(cos, pos, top_k=request.top_k)
    
    # 2. CO-PSO Mappings (if provided)
    co_pso_results = compute_similarity(cos, psos, top_k=request.top_k) if psos else []

    # Assemble CO-PO (and optionally PSO) Matrix
    matrix = {}
    table = []
    
    combined_po_psos = pos + psos
    target_ids = [p["id"] for p in combined_po_psos]

    for i, co in enumerate(cos):
        co_id = co["id"]
        matrix[co_id] = {}
        row = {"co_id": co_id, "co_text": co["text"]}
        
        # Initialize all to 0
        for tid in target_ids:
            matrix[co_id][tid] = 0
            row[tid] = 0

        # Fill POs
        for cand in co_po_results[i]["candidates"]:
            matrix[co_id][cand["po_id"]] = cand["level"]
            row[cand["po_id"]] = cand["level"]

        # Fill PSOs
        if psos:
            for cand in co_pso_results[i]["candidates"]:
                target_id = cand.get("po_id")
                if target_id:
                    matrix[co_id][target_id] = cand["level"]
                    row[target_id] = cand["level"]

        table.append(row)

    # 3. PO-PEO Mappings (if provided)
    peo_matrix = None
    peo_table = []
    if peos:
        po_peo_results = compute_similarity(pos, peos, top_k=request.top_k)
        peo_matrix = {}
        for i, po in enumerate(pos):
            po_id = po["id"]
            peo_matrix[po_id] = {}
            peo_row = {"po_id": po_id, "po_text": po["text"]}
            
            for peo in peos:
                peo_matrix[po_id][peo["id"]] = 0
                peo_row[peo["id"]] = 0
                
            for cand in po_peo_results[i]["candidates"]:
                target_id = cand.get("po_id")
                if target_id:
                    peo_matrix[po_id][target_id] = cand["level"]
                    peo_row[target_id] = cand["level"]
            
            peo_table.append(peo_row)

    return {
        "po_ids": [p["id"] for p in pos],
        "pso_ids": [p["id"] for p in psos],
        "peo_ids": [p["id"] for p in peos],
        "co_ids": [c["id"] for c in cos],
        "matrix": matrix,
        "table": table,
        "peo_matrix": peo_matrix,
        "peo_table": peo_table
    }

@app.get("/evaluate")
def evaluate_system():
    """
    Evaluates the mapping system using labeled_pairs.json.
    Returns precision@1 and precision@3 with full details.
    This endpoint may take 30-60 seconds on first run
    because it processes all labeled pairs.
    """
    results = evaluate_precision_at_k(k=3)
    return results




from api.pdf_services import generate_accreditation_pdf

@app.post("/export/pdf")
def export_pdf(payload: str = Form(...), file: UploadFile = File(None)):
    import json
    try:
        request_data = json.loads(payload)
    except:
        return {"error": "Invalid payload"}
        
    return generate_accreditation_pdf(request_data, file)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from mapping.similarity import compute_similarity, score_one_pair
from sequencer.topo_sort import build_semester_plan
from mapping.similarity import model, similarity_to_level
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from mapping.evaluator import evaluate_precision_at_k
from fastapi.middleware.cors import CORSMiddleware
from api.pdf_services import generate_accreditation_pdf
from fastapi.staticfiles import StaticFiles
import os

from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
from api.database import SessionLocal, engine, Base, get_db
import api.models as models

# Initialize and migrate SQLite Database tables/columns dynamically
Base.metadata.create_all(bind=engine)
try:
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("projects")]
    with engine.connect() as conn:
        modified = False
        if "matrix_json" not in columns:
            conn.execute(text("ALTER TABLE projects ADD COLUMN matrix_json TEXT DEFAULT '{}'"))
            modified = True
        if "peo_matrix_json" not in columns:
            conn.execute(text("ALTER TABLE projects ADD COLUMN peo_matrix_json TEXT DEFAULT '{}'"))
            modified = True
        if "courses_json" not in columns:
            conn.execute(text("ALTER TABLE projects ADD COLUMN courses_json TEXT DEFAULT '[]'"))
            modified = True
        if "sequencer_plan_json" not in columns:
            conn.execute(text("ALTER TABLE projects ADD COLUMN sequencer_plan_json TEXT DEFAULT '{}'"))
            modified = True
        if "attainment_settings_json" not in columns:
            conn.execute(text("ALTER TABLE projects ADD COLUMN attainment_settings_json TEXT DEFAULT '{}'"))
            modified = True
        if "student_marks_json" not in columns:
            conn.execute(text("ALTER TABLE projects ADD COLUMN student_marks_json TEXT DEFAULT '[]'"))
            modified = True
        if "co_attainment_json" not in columns:
            conn.execute(text("ALTER TABLE projects ADD COLUMN co_attainment_json TEXT DEFAULT '{}'"))
            modified = True
        if "po_attainment_json" not in columns:
            conn.execute(text("ALTER TABLE projects ADD COLUMN po_attainment_json TEXT DEFAULT '{}'"))
            modified = True
        if modified:
            conn.commit()
except Exception as e:
    print(f"Database Migration Warning: {e}")

app = FastAPI(title="Group 2 CO-PO Mapping API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static files of the workspace
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/static", StaticFiles(directory=workspace_dir), name="static")

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
    explanations = {}
    
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
            explanations[f"{co_id}_{cand['po_id']}"] = cand.get("explanation", "No justification available.")

        # Fill PSOs
        if psos:
            for cand in co_pso_results[i]["candidates"]:
                target_id = cand.get("po_id")
                if target_id:
                    matrix[co_id][target_id] = cand["level"]
                    row[target_id] = cand["level"]
                    explanations[f"{co_id}_{target_id}"] = cand.get("explanation", "No justification available.")

        table.append(row)

    # 3. PO-PEO Mappings (if provided)
    peo_matrix = None
    peo_table = []
    peo_explanations = {}
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
                    peo_explanations[f"{po_id}_{target_id}"] = cand.get("explanation", "No justification available.")
            
            peo_table.append(peo_row)

    return {
        "po_ids": [p["id"] for p in pos],
        "pso_ids": [p["id"] for p in psos],
        "peo_ids": [p["id"] for p in peos],
        "co_ids": [c["id"] for c in cos],
        "matrix": matrix,
        "table": table,
        "peo_matrix": peo_matrix,
        "peo_table": peo_table,
        "explanations": explanations,
        "peo_explanations": peo_explanations
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

# ----------- PYDANTIC SCHEMAS FOR PROJECTS -----------
class ProjectCreate(BaseModel):
    name: str

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    pos_json: Optional[str] = None
    psos_json: Optional[str] = None
    peos_json: Optional[str] = None
    matrix_json: Optional[str] = None
    peo_matrix_json: Optional[str] = None
    courses_json: Optional[str] = None
    sequencer_plan_json: Optional[str] = None
    attainment_settings_json: Optional[str] = None
    student_marks_json: Optional[str] = None
    co_attainment_json: Optional[str] = None
    po_attainment_json: Optional[str] = None

# ----------- DATABASE CRUD ENDPOINTS -----------
@app.get("/projects")
def list_projects(db: Session = Depends(get_db)):
    projects = db.query(models.Project).all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "created_at": p.created_at
        }
        for p in projects
    ]

@app.post("/projects")
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    db_project = models.Project(name=project.name)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return {"id": db_project.id, "name": db_project.name}

@app.get("/projects/{project_id}")
def get_project(project_id: int, db: Session = Depends(get_db)):
    db_project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        "id": db_project.id,
        "name": db_project.name,
        "created_at": db_project.created_at,
        "pos_json": db_project.pos_json,
        "psos_json": db_project.psos_json,
        "peos_json": db_project.peos_json,
        "matrix_json": db_project.matrix_json,
        "peo_matrix_json": db_project.peo_matrix_json,
        "courses_json": db_project.courses_json,
        "sequencer_plan_json": db_project.sequencer_plan_json,
        "attainment_settings_json": db_project.attainment_settings_json,
        "student_marks_json": db_project.student_marks_json,
        "co_attainment_json": db_project.co_attainment_json,
        "po_attainment_json": db_project.po_attainment_json
    }

@app.put("/projects/{project_id}")
def update_project(project_id: int, project: ProjectUpdate, db: Session = Depends(get_db)):
    db_project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    update_data = project.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_project, key, value)
        
    db.commit()
    db.refresh(db_project)
    return {"message": "Project updated successfully"}

@app.delete("/projects/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    db_project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete(db_project)
    db.commit()
    return {"message": "Project deleted successfully"}


# ----------- ATTAINMENT SCHEMAS & ENDPOINTS -----------

class AttainmentQuestion(BaseModel):
    id: str
    name: str
    max_marks: float
    co_id: str

class StudentMark(BaseModel):
    student_id: str
    student_name: str
    marks: dict # dict mapping question_id -> score (which can be a string or number)

class AttainmentRequest(BaseModel):
    cos: List[str]
    pos: List[str]
    psos: List[str] = []
    matrix: dict # co_id -> { target_id -> level }
    questions: List[AttainmentQuestion]
    students: List[StudentMark]
    target_score_percent: float = 60.0
    threshold_l1: float = 50.0
    threshold_l2: float = 60.0
    threshold_l3: float = 70.0

@app.post("/attainment/calculate")
def calculate_attainment(request: AttainmentRequest):
    # Step 1: Calculate question achievement stats
    question_stats = {}
    for q in request.questions:
        target_score = q.max_marks * (request.target_score_percent / 100.0)
        achieved_count = 0
        total_count = 0
        for s in request.students:
            val = s.marks.get(q.id)
            if val is not None and str(val).strip() != "":
                try:
                    val_float = float(val)
                    total_count += 1
                    if val_float >= target_score:
                        achieved_count += 1
                except ValueError:
                    pass
        
        pct = (achieved_count / total_count * 100.0) if total_count > 0 else 0.0
        question_stats[q.id] = {
            "id": q.id,
            "name": q.name,
            "co_id": q.co_id,
            "max_marks": q.max_marks,
            "achieved_count": achieved_count,
            "total_count": total_count,
            "percentage": round(pct, 2)
        }

    # Step 2: Calculate CO Attainment
    co_stats = {}
    for co_id in request.cos:
        # find all questions for this co_id
        q_ids = [q.id for q in request.questions if q.co_id == co_id]
        if not q_ids:
            co_stats[co_id] = {
                "co_id": co_id,
                "percentage": 0.0,
                "level": 0,
                "questions": []
            }
            continue
            
        co_pcts = [question_stats[qid]["percentage"] for qid in q_ids]
        avg_pct = sum(co_pcts) / len(co_pcts)
        
        # Determine level based on thresholds
        if avg_pct >= request.threshold_l3:
            level = 3
        elif avg_pct >= request.threshold_l2:
            level = 2
        elif avg_pct >= request.threshold_l1:
            level = 1
        else:
            level = 0
            
        co_stats[co_id] = {
            "co_id": co_id,
            "percentage": round(avg_pct, 2),
            "level": level,
            "questions": q_ids
        }

    # Step 3: Propagate to PO & PSO
    po_stats = {}
    pso_stats = {}
    
    # Calculate for POs
    for po_id in request.pos:
        weighted_sum = 0.0
        weight_sum = 0.0
        mapped_cos = []
        for co_id in request.cos:
            level_weight = request.matrix.get(co_id, {}).get(po_id, 0)
            if level_weight > 0:
                co_level = co_stats[co_id]["level"]
                weighted_sum += co_level * level_weight
                weight_sum += level_weight
                mapped_cos.append(co_id)
                
        attained_level = round(weighted_sum / weight_sum, 2) if weight_sum > 0 else 0.0
        po_stats[po_id] = {
            "po_id": po_id,
            "attainment": attained_level,
            "weight_sum": weight_sum,
            "mapped_cos": mapped_cos
        }

    # Calculate for PSOs
    for pso_id in request.psos:
        weighted_sum = 0.0
        weight_sum = 0.0
        mapped_cos = []
        for co_id in request.cos:
            level_weight = request.matrix.get(co_id, {}).get(pso_id, 0)
            if level_weight > 0:
                co_level = co_stats[co_id]["level"]
                weighted_sum += co_level * level_weight
                weight_sum += level_weight
                mapped_cos.append(co_id)
                
        attained_level = round(weighted_sum / weight_sum, 2) if weight_sum > 0 else 0.0
        pso_stats[pso_id] = {
            "pso_id": pso_id,
            "attainment": attained_level,
            "weight_sum": weight_sum,
            "mapped_cos": mapped_cos
        }

    return {
        "question_stats": question_stats,
        "co_stats": co_stats,
        "po_stats": po_stats,
        "pso_stats": pso_stats
    }

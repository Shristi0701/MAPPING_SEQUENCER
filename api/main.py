from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import List
from mapping.similarity import compute_similarity, score_one_pair
from sequencer.topo_sort import build_semester_plan
from mapping.similarity import model, similarity_to_level
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from mapping.evaluator import evaluate_precision_at_k
from fastapi.middleware.cors import CORSMiddleware

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




@app.post("/export/pdf")
def export_pdf(payload: str = Form(...), file: UploadFile = File(None)):
    import json
    import tempfile
    import io
    from fastapi.responses import FileResponse

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        return {"error": "reportlab/pypdf not installed."}

    try:
        request_data = json.loads(payload)
    except:
        return {"error": "Invalid payload"}

    cos = request_data.get("cos", [])
    pos = request_data.get("pos", [])
    psos = request_data.get("psos", [])
    peos = request_data.get("peos", [])
    top_k = request_data.get("top_k", 3)
    subject = request_data.get("subject", "")
    semester = request_data.get("semester", "")

    # 1. Compute Mappings
    co_po_results = compute_similarity(cos, pos, top_k=top_k)
    co_pso_results = compute_similarity(cos, psos, top_k=top_k) if psos else []
    po_peo_results = compute_similarity(pos, peos, top_k=top_k) if peos else []

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    title_style.alignment = 1 # Center
    title_style.fontSize = 16
    title_style.spaceAfter = 12

    normal_style = styles["Normal"]
    normal_style.fontSize = 11
    normal_style.leading = 14

    small_style = styles["Normal"]
    small_style.fontSize = 9
    small_style.leading = 12

    subtitle_style = styles["Heading3"]
    subtitle_style.fontSize = 13
    subtitle_style.leading = 16

    pdf_level_bg = {0: colors.white, 1: colors.HexColor('#FEF3CD'), 2: colors.HexColor('#D6EAF8'), 3: colors.HexColor('#D5F5E3')}
    pdf_level_fg = {0: colors.HexColor('#AAAAAA'), 1: colors.HexColor('#856404'), 2: colors.HexColor('#1A5276'), 3: colors.HexColor('#145A32')}

    elements = []
    
    # ---------------- SECTION 1: CO x PO + PSO MATRIX ----------------
    elements.append(Paragraph("<b>CO × PO & PSO MAPPING MATRIX</b>", title_style))
    elements.append(Paragraph(f"<b>Subject:</b> {subject} &nbsp;&nbsp;&nbsp;&nbsp; <b>Semester:</b> {semester}", subtitle_style))
    elements.append(Spacer(1, 8))
    
    legend_html = "<b>Legend:</b> &nbsp;&nbsp; <b>-</b> : No mapping &nbsp;&nbsp;|&nbsp;&nbsp; <b>1</b> : Low &nbsp;&nbsp;|&nbsp;&nbsp; <b>2</b> : Medium &nbsp;&nbsp;|&nbsp;&nbsp; <b>3</b> : High"
    elements.append(Paragraph(legend_html, normal_style))
    elements.append(Spacer(1, 12))

    target_cols = [p["id"] for p in pos] + [p["id"] for p in psos]
    headers = ["CO ID", "CO Text"] + target_cols
    data = [headers]
    
    matrix_data = [] # row levels for styling
    for i, co in enumerate(cos):
        row = [co["id"], Paragraph(co["text"], small_style)]
        row_lvls = [None, None]
        
        # POs
        po_map = {c["po_id"]: c["level"] for c in co_po_results[i]["candidates"]}
        for p in pos:
            lvl = po_map.get(p["id"], 0)
            row.append(str(lvl) if lvl > 0 else "-")
            row_lvls.append(lvl)
            
        # PSOs
        pso_map = {c["po_id"]: c["level"] for c in co_pso_results[i]["candidates"]} if psos else {}
        for ps in psos:
            lvl = pso_map.get(ps["id"], 0)
            row.append(str(lvl) if lvl > 0 else "-")
            row_lvls.append(lvl)
            
        data.append(row)
        matrix_data.append(row_lvls)

    col_widths = [45, 260] + [25] * len(target_cols)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    
    style_cmds = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A3A5C')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#DDDDDD')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (1,1), (1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,1), (-1,-1), 5),
        ('TOPPADDING', (0,1), (-1,-1), 5),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
    ]
    for r, row_lvls in enumerate(matrix_data):
        for c, lvl in enumerate(row_lvls):
            if lvl is not None:
                style_cmds.append(('BACKGROUND', (c, r+1), (c, r+1), pdf_level_bg.get(lvl, colors.white)))
                style_cmds.append(('TEXTCOLOR', (c, r+1), (c, r+1), pdf_level_fg.get(lvl, colors.black)))

    t.setStyle(TableStyle(style_cmds))
    elements.append(t)

    # ---------------- SECTION 3: PO x PEO MATRIX ----------------
    if peos:
        elements.append(PageBreak())
        elements.append(Paragraph("<b>PO × PEO MAPPING MATRIX</b>", title_style))
        elements.append(Spacer(1, 12))
        
        peo_headers = ["PO ID", "PO Text"] + [pe["id"] for pe in peos]
        peo_data = [peo_headers]
        peo_lvls_track = []
        
        for i, po in enumerate(pos):
            row = [po["id"], Paragraph(po["text"], small_style)]
            row_lvls = [None, None]
            mapping = {c["po_id"]: c["level"] for c in po_peo_results[i]["candidates"]}
            for pe in peos:
                lvl = mapping.get(pe["id"], 0)
                row.append(str(lvl) if lvl > 0 else "-")
                row_lvls.append(lvl)
            peo_data.append(row)
            peo_lvls_track.append(row_lvls)

        pt = Table(peo_data, colWidths=[45, 260] + [35]*len(peos))
        
        peo_style_cmds = [
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A3A5C')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('TOPPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#DDDDDD')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('ALIGN', (1,1), (1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,1), (-1,-1), 5),
            ('TOPPADDING', (0,1), (-1,-1), 5),
            ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ]
        for r, row_lvls in enumerate(peo_lvls_track):
            for c, lvl in enumerate(row_lvls):
                if lvl is not None:
                    peo_style_cmds.append(('BACKGROUND', (c, r+1), (c, r+1), pdf_level_bg.get(lvl, colors.white)))
                    peo_style_cmds.append(('TEXTCOLOR', (c, r+1), (c, r+1), pdf_level_fg.get(lvl, colors.black)))
                    
        pt.setStyle(TableStyle(peo_style_cmds))
        elements.append(pt)

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(letter), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    doc.build(elements)
    pdf_buffer.seek(0)

    writer = PdfWriter()
    if file:
        try:
            existing_pdf = PdfReader(file.file)
            for page in existing_pdf.pages: writer.add_page(page)
        except: pass
            
    new_pdf = PdfReader(pdf_buffer)
    for page in new_pdf.pages:
        writer.add_page(page)
        
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    writer.write(tmp_pdf.name)
    tmp_pdf.close()
    return FileResponse(tmp_pdf.name, media_type="application/pdf", filename="Accreditation_Report.pdf")

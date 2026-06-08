import json
import tempfile
import io
from fastapi.responses import FileResponse
from mapping.similarity import compute_similarity

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from pypdf import PdfReader, PdfWriter
    HAS_PDF_LIBS = True
except ImportError:
    HAS_PDF_LIBS = False

def generate_accreditation_pdf(request_data: dict, file) -> FileResponse:
    if not HAS_PDF_LIBS:
        return {"error": "reportlab/pypdf not installed."}

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

    # 1.1 Custom Overrides Lookup
    override_matrix = request_data.get("matrix")
    override_peo_matrix = request_data.get("peo_matrix")

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
            if override_matrix and co["id"] in override_matrix and p["id"] in override_matrix[co["id"]]:
                lvl = override_matrix[co["id"]][p["id"]]
            row.append(str(lvl) if lvl > 0 else "-")
            row_lvls.append(lvl)
            
        # PSOs
        pso_map = {c["po_id"]: c["level"] for c in co_pso_results[i]["candidates"]} if psos else {}
        for ps in psos:
            lvl = pso_map.get(ps["id"], 0)
            if override_matrix and co["id"] in override_matrix and ps["id"] in override_matrix[co["id"]]:
                lvl = override_matrix[co["id"]][ps["id"]]
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
                if override_peo_matrix and po["id"] in override_peo_matrix and pe["id"] in override_peo_matrix[po["id"]]:
                    lvl = override_peo_matrix[po["id"]][pe["id"]]
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

    # ---------------- SECTION 4: CO & PO ATTAINMENT METRICS ----------------
    co_attainment = request_data.get("co_attainment")
    po_attainment = request_data.get("po_attainment")
    pso_attainment = request_data.get("pso_attainment")

    # If they are passed as JSON strings, parse them
    if isinstance(co_attainment, str):
        try: co_attainment = json.loads(co_attainment)
        except: co_attainment = None
    if isinstance(po_attainment, str):
        try: po_attainment = json.loads(po_attainment)
        except: po_attainment = None
    if isinstance(pso_attainment, str):
        try: pso_attainment = json.loads(pso_attainment)
        except: pso_attainment = None

    if co_attainment and po_attainment:
        elements.append(PageBreak())
        elements.append(Paragraph("<b>CO & PO ATTAINMENT METRICS</b>", title_style))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("Direct assessment attainment levels computed from student exam performance and course mappings.", normal_style))
        elements.append(Spacer(1, 12))
        
        # 4.1 CO Attainment Table
        elements.append(Paragraph("<b>Course Outcome (CO) Attainment</b>", subtitle_style))
        elements.append(Spacer(1, 6))
        
        co_headers = ["CO ID", "CO Text", "Mapped Questions", "Attainment %", "Attainment Level"]
        co_data = [co_headers]
        co_lvls_track = []
        
        for co in cos:
            co_id = co["id"]
            co_entry = co_attainment.get(co_id, {})
            pct = co_entry.get("percentage", 0.0)
            lvl = co_entry.get("level", 0)
            q_list = ", ".join(co_entry.get("questions", []))
            
            row = [
                co_id, 
                Paragraph(co["text"], small_style), 
                Paragraph(q_list if q_list else "None", small_style),
                f"{pct}%", 
                str(lvl)
            ]
            co_data.append(row)
            co_lvls_track.append(lvl)
            
        cot = Table(co_data, colWidths=[40, 240, 150, 70, 70])
        co_style_cmds = [
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A3A5C')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('TOPPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#DDDDDD')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('ALIGN', (1,1), (1,-1), 'LEFT'),
            ('ALIGN', (2,1), (2,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,1), (-1,-1), 5),
            ('TOPPADDING', (0,1), (-1,-1), 5),
            ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ]
        for r, lvl in enumerate(co_lvls_track):
            co_style_cmds.append(('BACKGROUND', (4, r+1), (4, r+1), pdf_level_bg.get(lvl, colors.white)))
            co_style_cmds.append(('TEXTCOLOR', (4, r+1), (4, r+1), pdf_level_fg.get(lvl, colors.black)))
            
        cot.setStyle(TableStyle(co_style_cmds))
        elements.append(cot)
        elements.append(Spacer(1, 16))
        
        # 4.2 PO Attainment Table
        elements.append(Paragraph("<b>Program Outcome (PO & PSO) Attainment</b>", subtitle_style))
        elements.append(Spacer(1, 6))
        
        po_headers = ["PO/PSO ID", "PO/PSO Text", "Mapped COs", "Attainment Level (out of 3.00)"]
        po_data = [po_headers]
        
        # Combine pos and psos
        combined_outcomes = [("PO", p) for p in pos] + [("PSO", ps) for ps in (psos or [])]
        
        for o_type, o in combined_outcomes:
            o_id = o["id"]
            if o_type == "PO":
                o_entry = po_attainment.get(o_id, {})
            else:
                o_entry = (pso_attainment or {}).get(o_id, {})
                if not o_entry and pso_attainment is None:
                    # fallback to check inside po_attainment if not separated
                    o_entry = po_attainment.get(o_id, {})
                    
            attainment_val = o_entry.get("attainment", 0.0)
            mapped_list = ", ".join(o_entry.get("mapped_cos", []))
            
            row = [
                o_id, 
                Paragraph(o["text"], small_style), 
                Paragraph(mapped_list if mapped_list else "None", small_style),
                f"{attainment_val:.2f}"
            ]
            po_data.append(row)
            
        pot = Table(po_data, colWidths=[60, 310, 120, 80])
        po_style_cmds = [
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A3A5C')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('TOPPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#DDDDDD')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('ALIGN', (1,1), (1,-1), 'LEFT'),
            ('ALIGN', (2,1), (2,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,1), (-1,-1), 5),
            ('TOPPADDING', (0,1), (-1,-1), 5),
            ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ]
        # Highlight attainment cell based on score
        for r in range(1, len(po_data)):
            try:
                score = float(po_data[r][3])
                if score >= 2.5: color_lvl = 3
                elif score >= 1.5: color_lvl = 2
                elif score >= 0.5: color_lvl = 1
                else: color_lvl = 0
                po_style_cmds.append(('BACKGROUND', (3, r), (3, r), pdf_level_bg.get(color_lvl, colors.white)))
                po_style_cmds.append(('TEXTCOLOR', (3, r), (3, r), pdf_level_fg.get(color_lvl, colors.black)))
            except:
                pass
                
        pot.setStyle(TableStyle(po_style_cmds))
        elements.append(pot)

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

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

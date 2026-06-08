# MAPPING_SEQUENCER

An end-to-end Outcome-Based Education (OBE) and NBA Accreditation toolkit for educational institutions. This system provides automated semantic mapping of course outcomes, intelligent course sequencing, and comprehensive PDF report generation for accreditation purposes.

## 🚀 Features

### 1. Semantic Mapping Engine (CO-PO / PSO / PEO)
- Automatically computes mapping levels (0-3: None, Low, Medium, High) between Course Outcomes (COs) and Program Outcomes (POs), Program Specific Outcomes (PSOs), and Program Educational Objectives (PEOs).
- Utilizes NLP-based semantic similarity to identify conceptual overlap and Bloom's Taxonomy action verbs.
- Calculates precision at K using built-in evaluation scripts against labeled datasets.

### 2. Topological Course Sequencer
- Generates structured semester plans by analyzing course prerequisites.
- Ensures total credits per semester remain within the specified maximum limits.
- Validates the curriculum structure to prevent cyclical prerequisites.

### 3. Accreditation Reporting (PDF Export)
- Exports high-quality, landscape PDF reports via `reportlab`.
- Generates color-coded matrices for CO × PO & PSO mapping and PO × PEO mapping.
- Highlights mapping levels with a clear legend for easy auditing by accreditation bodies (NBA/AICTE).

### 4. Interactive Frontend UI
- A modern, responsive web UI (`demo_ui.html`) to visualize mappings, upload outcome definitions, test the sequencer, and trigger PDF exports.

## 📁 Project Structure

```text
MAPPING_SEQUENCER/
├── api/
│   ├── main.py                     # FastAPI application and endpoints
│   └── ...                         # Additional API components
├── mapping/                        # Core semantic mapping & evaluation logic
├── sequencer/                      # Course sequencing and topological sort logic
├── schemas/                        # Pydantic schemas for data validation
├── data/                           # Labeled pairs, raw COs, and dataset storage
├── demo_ui.html                    # Frontend interface for the system
├── process_syllabus.py             # Script to parse and process existing syllabi
├── perform_accreditation_mapping.py# CLI tool for mapping tasks
└── requirements.txt                # Python dependencies
```

## 🛠️ Installation & Setup

1. **Clone or Download the Repository**
2. **Set up a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you also install `reportlab` and `pypdf` if you intend to use the PDF export functionality).*

## 🚦 Running the System

### 1. Start the API Server
Run the FastAPI backend using Uvicorn:
```bash
uvicorn api.main:app --reload
```
The API will be accessible at `http://127.0.0.1:8000`. You can view the Swagger UI documentation at `http://127.0.0.1:8000/docs`.

### 2. Launch the Web UI
Open `demo_ui.html` directly in your web browser. Ensure the API server is running in the background for the interface to function properly.

## 🔗 Key Endpoints

- `POST /map/auto`: Maps provided COs to POs/PSOs/PEOs and returns similarities.
- `POST /map/matrix`: Generates the complete CO × (PO+PSO) and PO × PEO matrices.
- `POST /sequencer/plan`: Accepts a list of courses and prerequisites, returning a sequenced semester plan.
- `POST /export/pdf`: Accepts matrix data and outputs a formatted PDF file.
- `GET /evaluate`: Evaluates the mapping system's accuracy against `labeled_pairs.json`.

## 📄 License
This project is for educational and accreditation purposes.

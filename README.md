<<<<<<< HEAD
# AI-Based Syllabus Generator

An AI-powered command-line tool that interactively generates detailed academic syllabi compliant with NBA (National Board of Accreditation) and OBE (Outcome-Based Education) standards. Powered by the **llama3.1:8b** model via **Ollama**, this tool produces structured, professional syllabus documents and exports them directly to `.docx` format.

## Features

- **Interactive Generation**: Interactively prompts you to generate and confirm sections of the syllabus.
- **NBA/OBE Compliant**: Automatically maps generated content to Outcome-Based Education (OBE) norms, using Bloom's Taxonomy for course outcomes.
- **Comprehensive Output**: Generates:
  - Course Objectives
  - Progressive Unit Titles (Units I - V)
  - Detailed Unit Content/Subtopics
  - Measurable Course Outcomes (COs)
  - Suggested Textbooks & Reference Books
  - Relevant YouTube/NPTEL Resources
- **Automated Word Export**: Automatically compiles and formats the generated syllabus into a professionally styled Microsoft Word document (`.docx`).

## Prerequisites

Before running the script, ensure you have the following installed:

1. **Python 3.7+**
2. **Ollama**: You must have Ollama installed and running on your machine.
   - [Download Ollama](https://ollama.com/)
   - Pull the required model:
     ```bash
     ollama run llama3.1:8b
     ```
3. **python-docx**: The Python library used for creating Word documents.
   - Install via pip:
     ```bash
     pip install python-docx
     ```

## Usage

1. **Start Ollama**: Ensure the Ollama service is running in the background. If it isn't running by default, start it by running `ollama serve` in a terminal.
2. **Run the Script**:
   Execute the Python script in your terminal:
   ```bash
   python "syllabus_generator (4) (1).py"
   ```
3. **Follow the Prompts**:
   - Enter the Programme Level (e.g., Undergraduate).
   - Enter the Programme Name (e.g., B.Tech CSE).
   - Enter the Course Name (e.g., Artificial Intelligence).
4. **Review & Confirm**: The tool will generate sections one by one. You can review the output and type `yes` to accept or `no` to regenerate.
5. **Output**: Once completed, the tool will compile the data and save a formatted `.docx` file in your current directory (e.g., `Artificial_Intelligence_syllabus.docx`).

## Configuration

By default, the script connects to Ollama on `127.0.0.1:11434`. You can override this by setting the following environment variables:
- `OLLAMA_HOST` (default: `127.0.0.1`)
- `OLLAMA_PORT` (default: `11434`)

## License
*Feel free to add your own license information here.*
=======
# MAPPING_SEQUENCER

An end-to-end Outcome-Based Education (OBE) and NBA Accreditation toolkit for educational institutions. This system provides automated semantic mapping of course outcomes, intelligent course sequencing, AI-based syllabus generation, and comprehensive PDF report generation for accreditation purposes.

## 🚀 Features

### 1. Semantic Mapping Engine (CO-PO / PSO / PEO)
- Automatically computes mapping levels (0-3: None, Low, Medium, High) between Course Outcomes (COs) and Program Outcomes (POs), Program Specific Outcomes (PSOs), and Program Educational Objectives (PEOs).
- Utilizes NLP-based semantic similarity to identify conceptual overlap and Bloom's Taxonomy action verbs.
- Calculates precision at K using built-in evaluation scripts against labeled datasets.

### 2. Topological Course Sequencer
- Generates structured semester plans by analyzing course prerequisites.
- Ensures total credits per semester remain within the specified maximum limits.
- Validates the curriculum structure to prevent cyclical prerequisites.

### 3. AI-Based Syllabus Generator
- Located in the `Syllabus_generator/` directory.
- Powered by `llama3.1:8b` via Ollama to generate detailed academic syllabi.
- Interactively creates Course Objectives, Unit Titles, Unit Content, Course Outcomes (COs), Textbooks, Reference Books, and YouTube resources.
- Automatically exports the generated syllabus to a styled `.docx` format.

### 4. Accreditation Reporting (PDF Export)
- Exports high-quality, landscape PDF reports via `reportlab`.
- Generates color-coded matrices for CO × PO & PSO mapping and PO × PEO mapping.
- Highlights mapping levels with a clear legend for easy auditing by accreditation bodies (NBA/AICTE).

### 5. Interactive Frontend UI
- A modern, responsive web UI (`demo_ui.html`) to visualize mappings, upload outcome definitions, test the sequencer, and trigger PDF exports.

## 📁 Project Structure

```text
MAPPING_SEQUENCER/
├── api/
│   ├── main.py                     # FastAPI application and endpoints
│   └── ...                         # Additional API components
├── mapping/                        # Core semantic mapping & evaluation logic
├── sequencer/                      # Course sequencing and topological sort logic
├── Syllabus_generator/             # AI-based DOCX syllabus generation module
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
4. **Ollama Setup (For Syllabus Generator only)**
   - Download and install [Ollama](https://ollama.com/).
   - Pull the Llama 3.1 model: `ollama run llama3.1:8b`.

## 🚦 Running the System

### 1. Start the API Server
Run the FastAPI backend using Uvicorn:
```bash
uvicorn api.main:app --reload
```
The API will be accessible at `http://127.0.0.1:8000`. You can view the Swagger UI documentation at `http://127.0.0.1:8000/docs`.

### 2. Launch the Web UI
Open `demo_ui.html` directly in your web browser. Ensure the API server is running in the background for the interface to function properly.

### 3. Generate a Syllabus
Navigate to the `Syllabus_generator` directory and run the interactive script:
```bash
cd Syllabus_generator
python "syllabus_generator (4) (1).py"
```

## 🔗 Key Endpoints

- `POST /map/auto`: Maps provided COs to POs/PSOs/PEOs and returns similarities.
- `POST /map/matrix`: Generates the complete CO × (PO+PSO) and PO × PEO matrices.
- `POST /sequencer/plan`: Accepts a list of courses and prerequisites, returning a sequenced semester plan.
- `POST /export/pdf`: Accepts matrix data and outputs a formatted PDF file.
- `GET /evaluate`: Evaluates the mapping system's accuracy against `labeled_pairs.json`.

## 📄 License
This project is for educational and accreditation purposes.
>>>>>>> 7fe5ff03 (updated code)

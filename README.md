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

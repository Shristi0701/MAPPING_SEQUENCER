import requests
import json

def test_pdf_export():
    API_URL = "http://127.0.0.1:8000/export/pdf"
    
    payload = {
        "subject": "Computer Science & Engineering",
        "semester": "V",
        "cos": [
            {"id": "CO1", "text": "Illustrate the flowchart and design algorithm for a given problem and to develop C programs using operators"},
            {"id": "CO2", "text": "Develop conditional and iterative statements to write C programs"},
            {"id": "CO3", "text": "Exercise user defined functions to solve real time problems"}
        ],
        "pos": [
            {"id": "PO1", "text": "Engineering Knowledge: Apply the knowledge of mathematics, science, engineering fundamentals..."},
            {"id": "PO2", "text": "Problem Analysis: Identify, formulate, review research literature..."},
            {"id": "PO3", "text": "Design/Development of Solutions: Design solutions for complex engineering problems..."}
        ],
        "psos": [
            {"id": "PSO1", "text": "To design and develop the computer applications using the software and hardware concepts of the computer science."},
            {"id": "PSO2", "text": "To cultivate the skills for career, research, development and entrepreneurship."}
        ],
        "peos": [
            {"id": "PEO1", "text": "Students will gain the ability to identify, formulate, and solve challenging IT problems."},
            {"id": "PEO2", "text": "Students will develop professional skills that will prepare them for immediate employment..."}
        ],
        "top_k": 3
    }
    
    files = {
        'payload': (None, json.dumps(payload))
    }
    
    try:
        print("Sending request to generate PDF...")
        response = requests.post(API_URL, files=files)
        if response.status_code == 200:
            with open("Accreditation_Sample_Report.pdf", "wb") as f:
                f.write(response.content)
            print("Successfully generated Accreditation_Sample_Report.pdf")
        else:
            print(f"Failed to generate PDF: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error connecting to API: {e}")

if __name__ == "__main__":
    test_pdf_export()

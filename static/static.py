
from pathlib import Path
from docx2pdf import convert

# Locate the project root
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# PDF destination
STATIC_DIR = Path(__file__).resolve().parent

STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Search the entire project for DOCX files
docx_files = [
    file for file in PROJECT_DIR.rglob("*.docx")
    if ".venv" not in file.parts
    and "static" not in file.parts
    and "__pycache__" not in file.parts
]

print(f"Project directory: {PROJECT_DIR}")
print(f"Word documents found: {len(docx_files)}")

if not docx_files:
    print("No Word documents found. Check where extract.py saves them.")

else:
    for docx_file in docx_files:
        try:
            pdf_filename = docx_file.stem + ".pdf"
            pdf_path = STATIC_DIR / pdf_filename

            print(f"Converting: {docx_file}")

            convert(str(docx_file), str(pdf_path))

            if pdf_path.exists():
                print(f"Saved PDF: {pdf_path}")
            else:
                print(f"Conversion did not produce: {pdf_filename}")

        except Exception as e:
            print(f"Error converting {docx_file.name}: {e}")

print("\nPDF conversion process finished!")


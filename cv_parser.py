import os
from typing import Optional
import pdfplumber
import docx
def extract_cv_text(file_path: str) -> Optional[str]:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() or '' for page in pdf.pages)
            return text.strip()
        elif ext in ['.docx', '.doc']:
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        elif ext in ['.txt']:
            with open(file_path, encoding='utf-8') as f:
                return f.read().strip()
        else:
            print(f"Format de fichier non supporté : {ext}")
            return None
    except Exception as e:
        print(f"Erreur lors de l'extraction du texte du CV : {e}")
        return None 
import os
from typing import Optional
import openai

def summarize_cv_text(cv_text: str, openai_api_key: str) -> Optional[str]:
    if not cv_text or not openai_api_key:
        print("Texte du CV ou clé API manquante.")
        return None
    try:
        client = openai.Client(api_key=openai_api_key)
        prompt = (
            "Voici le texte d'un CV. Résume les informations essentielles sous forme structurée : "
            "- Compétences principales\n- Expériences professionnelles (poste, entreprise, durée, missions)\n- Diplômes et formations\n- Certifications (si présentes)\n- Autres informations pertinentes.\n"
            "Sois synthétique et clair.\n\nCV :\n" + cv_text
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"Erreur lors du résumé du CV : {e}")
        return None 
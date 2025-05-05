import streamlit as st
import os
from dotenv import load_dotenv
from cv_parser import extract_cv_text
from summarizer import summarize_cv_text
from process_jobs import configure_retrievers, run_sample_query, load_api_key, load_and_clean_data, get_chroma_collection
from process_jobs import OpenAIEmbeddings, chromadb
from langchain.vectorstores import Chroma

st.set_page_config(page_title="Matching CV & Offres d'emploi", page_icon="💼", layout="wide")
st.title("💼 Matching CV & Offres d'emploi")
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #0066cc; color: white;}
</style>
""", unsafe_allow_html=True)

# Chargement de la clé API
load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY', '')

# Chargement des données et du retriever (en cache pour rapidité)
@st.cache_resource(show_spinner=False)
def load_resources():
    job_data = load_and_clean_data('jobsspider_2024-01-23T04-34-07+00-00.csv')
    if job_data is None:
        st.error("Erreur lors du chargement des données des offres.")
        return None, None, None
        
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    if not openai_api_key:
        st.error("Clé API OpenAI manquante dans .env. Impossible d'initialiser les embeddings.")
        return None, None, None
    
    try:
        langchain_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        chroma_embedding_func = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-ada-002")
        job_collection = get_chroma_collection(chroma_client, "job", chroma_embedding_func)
        
        chroma_vector_store = Chroma(
            client=chroma_client,
            collection_name="job",
            embedding_function=langchain_embeddings
        )
        
        retriever = configure_retrievers(job_data, chroma_client, "job", langchain_embeddings, k=100)
        if retriever is None:
            st.error("Impossible de configurer le retriever hybride.")
            return None, None, None
        return retriever, job_data, chroma_vector_store
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation des ressources (embeddings/chroma): {e}")
        return None, None, None

def generate_cover_letter(cv_summary: str, job: dict, openai_api_key: str) -> str:
    import openai
    client = openai.Client(api_key=openai_api_key)
    job_description = job.get('job_description', '') or ''
    prompt = (
        "En te basant sur le résumé de CV suivant :\n" + cv_summary +
        "\n\nEt sur l'offre d'emploi suivante :\n" +
        f"Poste : {job.get('job_title', '')}\nEntreprise : {job.get('company_name', '')}\nLieu : {job.get('company_location', '')}\nDescription : {job_description}\n"
        "\nRédige une lettre de motivation personnalisée, professionnelle, concise (max 250 mots), adaptée à ce poste."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Erreur lors de la génération de la lettre : {e}"

st.header("1️⃣ Importez votre CV")
cv_file = st.file_uploader("Déposez votre CV (PDF, DOCX, TXT)", type=["pdf", "docx", "doc", "txt"])

if cv_file:
    if not os.path.exists("/tmp"):
        try:
            os.makedirs("/tmp")
        except OSError as e:
            st.error(f"Impossible de créer le répertoire /tmp: {e}")
            st.stop()
            
    temp_path = f"/tmp/{cv_file.name}"
    try:
        with open(temp_path, "wb") as f:
            f.write(cv_file.read())
        
        with st.spinner("Extraction du texte du CV..."):
            cv_text = extract_cv_text(temp_path)
            
    except Exception as e:
        st.error(f"Erreur lors de la gestion du fichier temporaire: {e}")
        cv_text = None
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                st.warning(f"Impossible de supprimer le fichier temporaire {temp_path}: {e}")

    if not cv_text:
        st.error("Impossible d'extraire le texte du CV.")
    else:
        st.success("Texte extrait du CV.")
        st.header("2️⃣ Résumé structuré du CV")
        if not openai_api_key:
            st.error("Clé API OpenAI manquante. Ajoutez-la dans votre .env.")
        else:
            with st.spinner("Résumé du CV en cours..."):
                summary = summarize_cv_text(cv_text, openai_api_key)
            if not summary:
                st.error("Erreur lors du résumé du CV.")
            else:
                st.info(summary)
                st.header("3️⃣ Offres d'emploi qui matchent votre profil")
                retriever, job_data, chroma_vector_store = load_resources()
                
                if retriever and chroma_vector_store:
                    with st.spinner("Recherche des offres d'emploi les plus pertinentes..."):
                        from process_jobs import EnsembleRetriever
                        results = retriever.get_relevant_documents(summary)
                        
                        result_ids = [doc.metadata.get('job_id') for doc in results if doc.metadata.get('job_id')]
                        st.write("IDs des résultats du retriever:", result_ids)
                        
                        scores = {}
                        score_results_debug = None
                        if result_ids:
                            try:
                                score_results = chroma_vector_store.similarity_search_with_relevance_scores(
                                    summary, 
                                    k=max(20, len(result_ids)),
                                )
                                score_results_debug = score_results
                                for doc_with_score, score_val in score_results:
                                    doc_id = doc_with_score.metadata.get('job_id')
                                    if doc_id in result_ids:
                                        scores[doc_id] = max(scores.get(doc_id, 0), score_val)
                            except Exception as e:
                                st.error(f"Erreur lors de la récupération des scores Chroma: {e}")
                                st.warning("Les scores de similarité n'ont pas pu être récupérés.")
                        
                        with st.expander("Infos de débogage des scores"):
                             st.write("Résultats bruts de similarity_search_with_relevance_scores:", score_results_debug)
                             st.write("Dictionnaire des scores construit:", scores)
                                
                    if not results:
                        st.warning("Aucune offre d'emploi trouvée.")
                    else:
                        st.markdown("--- Affichage des résultats ---")
                        for i, doc in enumerate(results):
                            meta = doc.metadata
                            doc_id = meta.get('job_id')
                            score = scores.get(doc_id) if doc_id else None 
                            
                            st.markdown(f"**{i+1}. {meta.get('job_title', 'Titre inconnu')}**")
                            st.markdown(f"   Entreprise : {meta.get('company_name', 'N/A')}")
                            st.markdown(f"   Lieu : {meta.get('company_location', 'N/A')}")
                            st.markdown(f"   (Debug ID: {doc_id}, Score Trouvé: {score})") 

                            if score is not None:
                                st.progress(float(score))
                                st.markdown(f"<span style='color: #6c757d; font-size: 0.9em;'>Score de similarité : {score:.3f}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown("<span style='color: #adb5bd; font-size: 0.9em;'>Score non disponible</span>", unsafe_allow_html=True)
                                
                            st.markdown(f"   [Voir l'offre]({meta.get('job_description_url', '#')})")
                            st.markdown(f"   <div style='color: #555; font-size: 0.95em;'>{doc.page_content[:250]}...</div>", unsafe_allow_html=True)
                            
                            with st.expander("Générer une lettre de motivation"):
                                btn_key = f"cover_{doc_id if doc_id else i}"
                                if st.button(f"Générer la lettre", key=btn_key):
                                    job_details_full = meta
                                    job_details_full['job_description'] = doc.page_content
                                    with st.spinner("Génération de la lettre..."):
                                        lettre = generate_cover_letter(summary, job_details_full, openai_api_key)
                                    if "Erreur" in lettre:
                                        st.error(lettre)
                                    else:
                                        st.success("Lettre générée :")
                                        st.text_area("Lettre", lettre, height=300)
                                        st.download_button(
                                            label="Télécharger la lettre",
                                            data=lettre,
                                            file_name=f"lettre_motivation_{meta.get('job_title', 'offre').replace(' ', '_')}_{i+1}.txt",
                                            mime="text/plain"
                                        )
                            st.markdown("---") 
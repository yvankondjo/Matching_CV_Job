import streamlit as st
import os
from dotenv import load_dotenv
from cv_parser import extract_cv_text
from summarizer import summarize_cv_text
from process_jobs import configure_retrievers, run_sample_query, load_api_key, load_and_clean_data, get_chroma_collection
from process_jobs import OpenAIEmbeddings, chromadb
from langchain.vectorstores import Chroma
import numpy as np # Pour calcul de similarité
from numpy.linalg import norm # Pour calcul de similarité

st.set_page_config(page_title="Matching CV & Job", page_icon="💼", layout="wide")
st.title("💼 Matching CV & Job")
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #0066cc; color: white;}
</style>
""", unsafe_allow_html=True)

# Chargement de la clé API
load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY', '')

# Initialisation de la session state pour stocker les lettres et résultats
if 'lettres_generees' not in st.session_state:
    st.session_state.lettres_generees = {}
if 'job_results' not in st.session_state:
    st.session_state.job_results = []
if 'query_embedding' not in st.session_state:
    st.session_state.query_embedding = None
if 'embeddings_map' not in st.session_state:
    st.session_state.embeddings_map = {}

# Chargement des données et du retriever (en cache pour rapidité)
@st.cache_resource(show_spinner=False)
def load_resources():
    job_data = load_and_clean_data('jobsspider_2024-01-23T04-34-07+00-00.csv')
    if job_data is None:
        st.error("Erreur lors du chargement des données des offres.")
        return None, None, None, None
        
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    if not openai_api_key:
        st.error("Clé API OpenAI manquante dans .env. Impossible d'initialiser les embeddings.")
        return None, None, None, None
    
    try:
        langchain_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        chroma_vector_store = Chroma(
            client=chroma_client,
            collection_name="job",
            embedding_function=langchain_embeddings
        )
        
        # Assurer que la collection existe (peut être fait par Chroma() mais vérifions)
        try: 
             chroma_client.get_collection("job")
        except: 
            st.warning("Collection 'job' non trouvée initialement, une nouvelle pourrait être créée par Chroma(). Assurez-vous que process_jobs a bien peuplé la base.")

        retriever = configure_retrievers(job_data, chroma_client, "job", langchain_embeddings, k=100)
        if retriever is None:
            st.error("Impossible de configurer le retriever hybride.")
            return None, None, None, None
            
        # Retourner aussi l'objet embeddings_model
        return retriever, job_data, chroma_vector_store, langchain_embeddings
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation des ressources (embeddings/chroma): {e}")
        return None, None, None, None

def calculate_cosine_similarity(emb1, emb2):
    """Calcule la similarité cosinus entre deux embeddings numpy."""
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)
    # Gérer le cas où les normes sont nulles pour éviter la division par zéro
    norm1 = norm(emb1)
    norm2 = norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    # La similarité cosinus est le produit scalaire divisé par le produit des normes
    # Note: Pour les embeddings OpenAI normalisés, le produit scalaire suffit.
    # Vérifions la normalisation (proche de 1)
    # print(f"Norm emb1: {norm1}, Norm emb2: {norm2}") 
    # return np.dot(emb1, emb2) # Si normalisé
    return np.dot(emb1, emb2) / (norm1 * norm2)

def generate_cover_letter(cv_summary: str, job: dict, openai_api_key: str) -> str:
    import openai
    if not cv_summary or not job or not openai_api_key:
         return "Erreur: Données manquantes pour générer la lettre (CV, offre ou clé API)."
    try:
        client = openai.Client(api_key=openai_api_key)
        job_description = job.get('job_description', '') or ''
        prompt = (
            "En te basant sur le résumé de CV suivant :\n" + cv_summary +
            "\n\nEt sur l'offre d'emploi suivante :\n" +
            f"Poste : {job.get('job_title', 'N/A')}\nEntreprise : {job.get('company_name', 'N/A')}\nLieu : {job.get('company_location', 'N/A')}\nDescription : {job_description}\n"
            "\nRédige une lettre de motivation personnalisée, professionnelle, concise (max 250 mots), adaptée à ce poste."
        )
        
        # Augmenter le nombre de tentatives max et le timeout
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3,
                    timeout=60  # 60 secondes de timeout
                )
                generated_text = response.choices[0].message.content.strip()
                return generated_text
            except Exception as retry_error:
                if attempt < max_retries - 1:
                    # Attendre un peu avant de réessayer
                    import time
                    time.sleep(2)
                    continue
                else:
                    raise retry_error
    except Exception as e:
        error_message = f"Erreur lors de la génération de la lettre via API OpenAI: {e}"
        print(error_message)
        return error_message

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
                st.header("3️⃣ Offres d'emploi correspondantes")
                retriever, job_data, chroma_vector_store, embeddings_model = load_resources()
                
                if retriever and chroma_vector_store and embeddings_model:
                    scores_by_id = {}
                    query_embedding = None
                    stored_embeddings_map = {} # Pour stocker {id: embedding}

                    # 1. Obtenir l'embedding du résumé (requête)
                    with st.spinner("Calcul de l'embedding du résumé..."):
                        if st.session_state.query_embedding is not None:
                            query_embedding = st.session_state.query_embedding
                        else:
                            try:
                                query_embedding = embeddings_model.embed_query(summary)
                                st.session_state.query_embedding = query_embedding
                            except Exception as e:
                                st.error(f"Erreur calcul embedding résumé: {e}")

                    # 2. Obtenir les résultats classés par le retriever hybride
                    with st.spinner("Recherche hybride des offres..."):
                        # Utiliser les résultats déjà stockés si disponibles
                        if len(st.session_state.job_results) > 0:
                            results = st.session_state.job_results
                        else:
                            try: 
                                results = retriever.get_relevant_documents(summary)
                                # Stocker dans la session state pour éviter de recalculer
                                st.session_state.job_results = results
                            except Exception as e: 
                                st.error(f"Erreur recherche hybride: {e}")
                                results = []
                                
                    # 3. Récupérer les embeddings stockés pour les résultats trouvés
                    if results and query_embedding:
                        result_ids = [doc.metadata.get('job_id') for doc in results if doc.metadata.get('job_id')]
                
                        
                        if result_ids:
                            with st.spinner("Récupération des embeddings stockés..."):
                                try:
                                    # Utiliser _collection.get pour les embeddings spécifiques
                                    stored_data = chroma_vector_store._collection.get(ids=result_ids, include=['embeddings'])
                                    
                                    # Vérifier et construire le mapping ID -> Embedding
                                    if stored_data and stored_data.get('ids'):
                                        embeddings_list = stored_data.get('embeddings')
                                        ids_list = stored_data.get('ids')
                                        if embeddings_list is not None and len(embeddings_list) > 0 and len(ids_list) == len(embeddings_list):
                                            for id_val, emb_val in zip(ids_list, embeddings_list):
                                                st.session_state.embeddings_map[id_val] = emb_val
                                            st.success(f"{len(st.session_state.embeddings_map)} embeddings stockés récupérés.") # Succès
                                        else:
                                            st.warning(f"Problème de cohérence ou embeddings vides récupérés: ids={len(ids_list)}, embs={len(embeddings_list) if embeddings_list else 0}")
                                except Exception as e:
                                    st.error(f"Impossible de récupérer les embeddings stockés via .get(): {e}") 
                    
                    # 4. Calculer et afficher les scores
                    if not results: st.warning("Aucune offre trouvée.")
                    else:
                        st.markdown("--- Affichage des résultats (classés par pertinence hybride) ---")
                        for i, doc in enumerate(results):
                            meta = doc.metadata
                            doc_id = meta.get('job_id')
                            score = None # Initialiser
                            
                            # Calculer le score si l'ID est trouvé dans notre map
                            if doc_id and query_embedding and doc_id in st.session_state.embeddings_map:
                                try:
                                    doc_embedding = st.session_state.embeddings_map[doc_id]
                                    score = calculate_cosine_similarity(query_embedding, doc_embedding)
                                except Exception as e: st.warning(f"Erreur calcul score ID {doc_id}: {e}")

                            st.markdown(f"**{i+1}. {meta.get('job_title', 'Titre inconnu')}**")
                            st.markdown(f"   Entreprise : {meta.get('company_name', 'N/A')}")
                            st.markdown(f"   Lieu : {meta.get('company_location', 'N/A')}")
                            
                            if score is not None:
                                st.progress(float(score))
                                st.markdown(f"<span style='color: #6c757d; font-size: 0.9em;'>Score similarité sémantique : {score:.3f}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown("<span style='color: #adb5bd; font-size: 0.9em;'>Score sémantique non calculé</span>", unsafe_allow_html=True)
                                
                            st.markdown(f"   [Voir l'offre]({meta.get('job_description_url', '#')})")
                            st.markdown(f"   <div style='color: #555; font-size: 0.95em;'>{doc.page_content[:250]}...</div>", unsafe_allow_html=True)
                            
                            with st.expander("Générer une lettre de motivation"):
                                btn_key = f"cover_{doc_id if doc_id else i}"
                                job_details_full = meta.copy()
                                job_details_full['job_description'] = doc.page_content
                                
                                # Vérifier si une lettre existe déjà pour cette offre
                                if doc_id in st.session_state.lettres_generees:
                                    lettre = st.session_state.lettres_generees[doc_id]
                                    st.success("Lettre générée :")
                                    st.text_area("Lettre", lettre, height=300, key=f"letter_text_{doc_id}")
                                    st.download_button(
                                        label="Télécharger la lettre",
                                        data=lettre,
                                        file_name=f"lettre_motivation_{meta.get('job_title', 'offre').replace(' ', '_')}_{i+1}.txt",
                                        mime="text/plain",
                                        key=f"dl_btn_{doc_id}"
                                    )
                                # Si pas de lettre existante, montrer le bouton de génération
                                elif st.button(f"Générer la lettre", key=btn_key):
                                    with st.spinner("Génération de la lettre..."):
                                        lettre = generate_cover_letter(summary, job_details_full, openai_api_key)
                                    
                                    if lettre and not lettre.startswith("Erreur"):
                                        # Stocker la lettre dans la session state
                                        st.session_state.lettres_generees[doc_id] = lettre
                                        st.success("Lettre générée :")
                                        st.text_area("Lettre", lettre, height=300, key=f"letter_text_{doc_id}")
                                        st.download_button(
                                            label="Télécharger la lettre",
                                            data=lettre,
                                            file_name=f"lettre_motivation_{meta.get('job_title', 'offre').replace(' ', '_')}_{i+1}.txt",
                                            mime="text/plain",
                                            key=f"dl_btn_{doc_id}"
                                        )
                                    else:
                                        error_msg = lettre if lettre else "La génération a échoué sans message d'erreur."
                                        st.error(error_msg)
                                        
                            st.markdown("---") 
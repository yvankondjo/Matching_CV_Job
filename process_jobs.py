import os
import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Any, Optional

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

def load_api_key() -> Optional[str]:
    """Charge la clé API OpenAI depuis les variables d'environnement."""
    _ = load_dotenv(find_dotenv())
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Erreur : Clé API OpenAI (OPENAI_API_KEY) non trouvée. "
              "Vérifiez votre fichier .env ou vos variables d'environnement.")
        return None
    return api_key

def get_openai_embedding_function(api_key: str) -> Optional[embedding_functions.OpenAIEmbeddingFunction]:
    """Crée et retourne l'objet fonction d'embedding OpenAI."""
    try:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-ada-002"
        )
        print("Fonction d'embedding OpenAI prête.")
        return openai_ef
    except Exception as e:
        print(f"Erreur lors de l'initialisation de la fonction d'embedding OpenAI: {e}")
        return None

def clean_text(text: Any, location: Any) -> str:
    """Nettoie le texte de la description de poste et ajoute la localisation."""
    text_str = str(text) if pd.notna(text) else ""
    location_str = str(location) if pd.notna(location) else ""

    text_1 = re.sub(r'\n|\r', ' ', text_str.lower())
    text_2 = re.sub(r'show (less|more)', '', text_1, flags=re.IGNORECASE)
    text_3 = re.sub(r'\(?https?://\S+\)?', '', text_2)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F" u"\U0001F780-\U0001F7FF" u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF" u"\U0001FA70-\U0001FAFF" u"\U00002700-\U000027BF"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text_4 = emoji_pattern.sub('', text_3)
    text_clean = re.sub(r'\s+', ' ', text_4).strip()
    text_final = f"{text_clean} {location_str}".strip()
    return text_final

def load_and_clean_data(file_path: str) -> Optional[pd.DataFrame]:
    """Charge les données depuis un CSV et applique le nettoyage."""
    try:
        data = pd.read_csv(file_path)
        print(f"Données chargées depuis {file_path}. Shape: {data.shape}")
        
        # Assurer l'existence des colonnes nécessaires
        required_cols = ['job_description', 'company_location', 'job_id']
        if not all(col in data.columns for col in required_cols):
            print(f"Erreur: Colonnes requises ({required_cols}) manquantes dans {file_path}")
            return None

        # Convertir en string pour éviter les erreurs de type
        data = data.astype(str)

        print("Nettoyage des descriptions de poste...")
        data['job_description_location'] = data.apply(
            lambda row: clean_text(row['job_description'], row['company_location']),
            axis=1
        )
        print("Nettoyage terminé.")
        
        # Afficher un exemple
        print("\nExemple après nettoyage (avec localisation) :")
        print(data['job_description_location'][0])
        
        return data
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement ou nettoyage des données : {e}")
        return None

def get_chroma_collection(client: chromadb.Client, 
                          collection_name: str, 
                          embedding_function: embedding_functions.OpenAIEmbeddingFunction
                          ) -> Optional[chromadb.Collection]:
    """Récupère ou crée une collection ChromaDB."""
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Collection '{collection_name}' obtenue ou créée. Nombre actuel d'éléments : {collection.count()}")
        return collection
    except Exception as e:
        print(f"Erreur lors de la création/obtention de la collection ChromaDB '{collection_name}' : {e}")
        return None

def build_vector_base(data: pd.DataFrame, collection: chromadb.Collection) -> None:
    """Ajoute les données du DataFrame à la collection ChromaDB par lots."""
    print(f"\nDébut de l'ajout/mise à jour de {len(data)} éléments dans la collection '{collection.name}'...")
    
    job_ids: List[str] = data['job_id'].astype(str).tolist()
    if len(job_ids) != len(set(job_ids)):
        print("Attention : Les job_id ne sont pas uniques. Les doublons seront écrasés lors de l'ajout.")

    documents: List[str] = data['job_description_location'].tolist()

    metadata_cols = ['company_name', 'company_location', 'job_title', 'job_description_url', 'time']
    valid_metadata_cols = [col for col in metadata_cols if col in data.columns]
    metadatas: List[Dict[str, Any]] = data[valid_metadata_cols].to_dict('records')

    batch_size = 100
    total_batches = (len(job_ids) + batch_size - 1) // batch_size

    for i in range(0, len(job_ids), batch_size):
        batch_ids = job_ids[i:min(i+batch_size, len(job_ids))]
        batch_documents = documents[i:min(i+batch_size, len(job_ids))]
        batch_metadatas = metadatas[i:min(i+batch_size, len(job_ids))]
        current_batch_num = (i // batch_size) + 1

        try:
            print(f"  Ajout du lot {current_batch_num}/{total_batches} ({len(batch_ids)} éléments)...", end='\r')
            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
        except Exception as e:
            print(f"\nErreur lors de l'ajout du lot {current_batch_num} (indices {i} à {i+batch_size}): {e}")
            # break # Décommenter pour arrêter en cas d'erreur sur un lot

    print(f"\nAjout terminé. Nombre total d'éléments dans la collection : {collection.count()}")

def configure_retrievers(data: pd.DataFrame, 
                         chroma_client: chromadb.Client, 
                         collection_name: str, 
                         langchain_embeddings,
                         k: int = 100
                         ) -> Optional[EnsembleRetriever]:
    """Configure les retrievers BM25, Chroma et l'EnsembleRetriever."""
    print("\nConfiguration des retrievers...")
    cleaned_docs_for_bm25 = data['job_description_location'].tolist()
    metadata_cols = ['company_name', 'company_location', 'job_title', 'job_description_url', 'time', 'job_id']
    valid_metadata_cols = [col for col in metadata_cols if col in data.columns]
    metadatas_for_bm25: List[Dict[str, Any]] = data[valid_metadata_cols].to_dict('records')
    langchain_docs = [Document(page_content=doc, metadata=meta) for doc, meta in zip(cleaned_docs_for_bm25, metadatas_for_bm25)]
    try:
        bm25_retriever = BM25Retriever.from_documents(langchain_docs)
        bm25_retriever.k = k
        print("Retriever BM25 configuré.")
    except Exception as e:
        print(f"Erreur lors de la configuration de BM25Retriever: {e}")
        bm25_retriever = None
    try:
        chroma_vector_store = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=langchain_embeddings
        )
        chroma_retriever = chroma_vector_store.as_retriever(search_kwargs={"k": k})
        print("Retriever Chroma configuré.")
    except Exception as e:
        print(f"Erreur lors de la configuration du retriever Chroma: {e}")
        chroma_retriever = None
    if bm25_retriever and chroma_retriever:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.1, 0.9]
        )
        print("Ensemble Retriever (BM25 + Chroma) configuré.")
        return ensemble_retriever
    else:
        print("Impossible de configurer EnsembleRetriever car un des retrievers de base a échoué.")
        return None

def run_sample_query(retriever: EnsembleRetriever, query: str, k: int = 100) -> None:
    """Exécute une requête de test avec l'EnsembleRetriever."""
    print(f"\n--- Test de la Recherche Hybride --- ")
    print(f"Requête : '{query}'")
    try:
        hybrid_results = retriever.get_relevant_documents(query)
        print(f"Résultats trouvés ({len(hybrid_results)}):\n")
        if not hybrid_results:
            print("Aucun document pertinent trouvé.")
        else:
            for i, doc in enumerate(hybrid_results[:k]):
                meta_info = {k: doc.metadata.get(k, 'N/A') for k in ['job_title', 'company_name', 'company_location']}
                print(f"  {i+1}. {meta_info.get('job_title')} - {meta_info.get('company_name')} ({meta_info.get('company_location')})")
                print(f"     Extrait: {doc.page_content[:200]}...\n")
    except Exception as e:
        print(f"Erreur lors de la recherche avec EnsembleRetriever: {e}")


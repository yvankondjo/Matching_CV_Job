# Matching CV-Job & Lettre de Motivation

Ce projet utilise l'intelligence artificielle pour aider à faire correspondre les CVs avec les offres d'emploi et générer des lettres de motivation personnalisées. Il extrait le texte des CVs (PDF, DOCX, TXT), résume le CV à l'aide de l'API OpenAI, et fait correspondre le résumé avec les offres d'emploi stockées dans une base de données ChromaDB.

## Fonctionnalités Principales
- **Extraction de Texte**: Extraction du texte des fichiers CV en formats PDF, DOCX, et TXT.
- **Résumé de CV**: Utilisation de l'API OpenAI pour générer un résumé concis du CV.
- **Correspondance avec Offres d'Emploi**: Utilisation d'un récupérateur hybride combinant BM25 et des embeddings sémantiques pour faire correspondre les résumés de CV avec les offres d'emploi.
- **Affichage des Scores de Similarité**: Affichage des scores de similarité pour chaque correspondance d'offre d'emploi.
- **Génération de Lettres de Motivation**: Génération de lettres de motivation personnalisées pour les offres d'emploi correspondantes.
- **Téléchargement de Lettres de Motivation**: Possibilité de télécharger les lettres de motivation générées.

## Aperçu de l'Application

<p align="center">
  <img src="images/Etape 1 & 2.png" alt="Import du CV et résumé" width="32%" />
  <img src="images/Etape 3.png" alt="Correspondance avec offres d'emploi" width="32%" />
  <img src="images/Lettre de motivation.png" alt="Génération de lettre de motivation" width="32%" />
</p>

## Installation et Utilisation
1. **Cloner le dépôt**: Clonez ce dépôt sur votre machine locale.
2. **Installer les dépendances**: Assurez-vous d'avoir Python installé, puis exécutez `pip install -r requirements.txt` pour installer les dépendances nécessaires.
3. **Configurer l'API OpenAI**: Ajoutez votre clé API OpenAI dans un fichier `.env` à la racine du projet.
4. **Lancer l'application**: Exécutez `streamlit run main.py` pour démarrer l'application Streamlit.

## Structure du Projet
- `main.py`: Application Streamlit pour gérer les téléchargements de CV, afficher les correspondances d'offres, et générer des lettres de motivation.
- `cv_parser.py`: Module pour l'extraction de texte des CVs.
- `summarizer.py`: Module pour résumer le texte des CVs.
- `process_jobs.py`: Module pour configurer les récupérateurs et gérer la base de données ChromaDB.

## Contribuer
Les contributions sont les bienvenues! Veuillez soumettre une pull request ou ouvrir une issue pour discuter des changements que vous souhaitez apporter.


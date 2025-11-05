import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import uvicorn

# === Définition des données d'entrée ===
class Query(BaseModel):
    text: str
    model_name: str = "intent_classifier"  # nom du modèle pré-entraîné

# === Création de l'application FastAPI ===
app = FastAPI(title="Intent Classifier Service")

# === Dossier pour stocker les modèles ===
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# === Cache pour modèles chargés ===
loaded_models = {}

# === Fonction pour charger un modèle existant uniquement ===
def load_model(model_name: str):
    if model_name in loaded_models:
        return loaded_models[model_name]

    classifier_file = os.path.join(MODELS_DIR, f"{model_name}.pkl")

    if not os.path.exists(classifier_file):
        raise FileNotFoundError(f"Le fichier {classifier_file} n'existe pas. Veuillez pré-entraîner le modèle avant d'utiliser le service.")

    with open(classifier_file, "rb") as f:
        label_embeddings = pickle.load(f)

    # Charger le modèle d'embedding
    model_embed = SentenceTransformer('all-MiniLM-L6-v2')
    loaded_models[model_name] = (model_embed, label_embeddings)
    return model_embed, label_embeddings

# === Fonction de classification ===
def classify(model, label_embeddings, text: str) -> str:
    query_emb = model.encode(text, convert_to_tensor=True)
    scores = {label: util.cos_sim(query_emb, emb).item() for label, emb in label_embeddings.items()}
    best_label = max(scores, key=scores.get)
    return best_label

# === Endpoint de classification uniquement ===
@app.post("/classify")
def classify_text(query: Query):
    try:
        model, label_embeddings = load_model(query.model_name)
        label = classify(model, label_embeddings, query.text)
        return {"label": label}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Lancement automatique du serveur Uvicorn ===
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )

import os
import json
import pickle
import argparse
from sentence_transformers import SentenceTransformer, util

# === Dossier pour stocker les modèles ===
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# === Fonctions ===

def train_classifier(model: SentenceTransformer, data: dict, save_path: str):
    """Entraîne le classifieur à partir des données et sauvegarde le résultat."""
    print("🔧 Entraînement du classifieur d’intentions...")
    label_embeddings = {
        label: model.encode(texts, convert_to_tensor=True).mean(dim=0)
        for label, texts in data.items()
    }
    with open(save_path, "wb") as f:
        pickle.dump(label_embeddings, f)
    print(f"✅ Classifieur entraîné et sauvegardé dans {save_path}")
    return label_embeddings


def load_or_train_classifier(config_path: str):
    """Charge la configuration depuis JSON et le classifieur si possible."""
    # Charger JSON
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = config.get("model_name", "intent_classifier")
    labels_data = config.get("labels", {})

    classifier_file = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    if os.path.exists(classifier_file):
        print(f"📦 Chargement du classifieur existant depuis {classifier_file}")
        with open(classifier_file, "rb") as f:
            label_embeddings = pickle.load(f)
    else:
        label_embeddings = train_classifier(model, labels_data, classifier_file)

    return model, label_embeddings, model_name


def classify(model, label_embeddings, text: str) -> str:
    """Classifie une phrase et retourne le label."""
    query_emb = model.encode(text, convert_to_tensor=True)
    scores = {label: util.cos_sim(query_emb, emb).item() for label, emb in label_embeddings.items()}
    best_label = max(scores, key=scores.get)
    return best_label


# === Interface CLI ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifieur d'intentions générique")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier JSON de configuration")
    parser.add_argument("--text", type=str, help="Phrase à classifier")
    args = parser.parse_args()

    model, label_embeddings, model_name = load_or_train_classifier(args.config)

    if args.text:
        label = classify(model, label_embeddings, args.text)
        print(label)
    else:
        print(f"\n💬 {model_name} prêt. Entrez des phrases à classifier (ou 'exit') :")
        while True:
            text = input("Utilisateur : ")
            if text.lower() in ["exit", "quit"]:
                break
            label = classify(model, label_embeddings, text)
            print(f"→ Label détecté : {label}\n")

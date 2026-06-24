"""Merge Fine-tuning and/or FT+RAG results into the main comparison file."""
import json

reports = r"C:\Users\pndia\OneDrive\Desktop\COUR SECOND SEMESTRE M2\COUR IA\Pratique\project-rag\reports"

# Load existing comparison
with open(f"{reports}/resultats_comparaison.json", encoding="utf-8") as f:
    comp = json.load(f)

def _merge_file(filepath, label):
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"  Ignore: {filepath}")
        return
    for r in data["resume"]:
        existing = [e for e in comp["resume"] if e["modele_approche"] == r["modele_approche"]]
        if existing:
            print(f"  Remplace: {r['modele_approche']}")
            comp["resume"] = [e for e in comp["resume"] if e["modele_approche"] != r["modele_approche"]]
        comp["resume"].append(r)
    for k, v in data["details"].items():
        comp["details"][k] = v
    print(f"  Fusionne: {label}")

print("Merge des resultats...")
_merge_file(f"{reports}/resultats_finetuning.json", "Fine-tuning")
_merge_file(f"{reports}/resultats_finetuning_rag.json", "Fine-tuning + RAG")

# Sort
order_key = {"LLM seul": 0, "RAG simple": 1, "RAG + Reranking": 2, "Hybrid Search": 3, "Fine-tuning": 4, "Fine-tuning + RAG": 5}
model_order = {"mistral-small": 0, "mistral-large": 1, "fine-tuned (Phi-3)": 2}
comp["resume"].sort(key=lambda r: (model_order.get(r.get("modele", "z"), 99), order_key.get(r.get("approche", "z"), 99)))

with open(f"{reports}/resultats_comparaison.json", "w", encoding="utf-8") as f:
    json.dump(comp, f, indent=2, ensure_ascii=True)

print(f"\nFusion complete: {len(comp['resume'])} entrees")
for r in comp["resume"]:
    print(f"  {r['modele_approche']:45s} EM={r['exact_match_rate']:<8} F1={r.get('f1_moyen', '?'):<8} n={r.get('nb_questions', '?')}")

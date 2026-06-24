"""
Fusionne les resultats du Fine-tuning (Colab) dans la comparaison locale.
Usage:
  python scripts/fusionner_resultats_ft.py chemin/vers/resultats_finetuning.json
"""
import json
import sys
import os

COMPARAISON_PATH = "reports/resultats_comparaison.json"
GRAPHIQUES_SCRIPT = "evaluation/graphiques.py"

def main(ft_path: str):
    if not os.path.isfile(ft_path):
        print(f"Fichier introuvable: {ft_path}")
        sys.exit(1)
    if not os.path.isfile(COMPARAISON_PATH):
        print(f"Fichier comparaison introuvable: {COMPARAISON_PATH}")
        print("Lance d'abord la comparaison via l'interface ou run_comparaison.py")
        sys.exit(1)

    with open(ft_path, encoding="utf-8") as f:
        ft_data = json.load(f)
    with open(COMPARAISON_PATH, encoding="utf-8") as f:
        comp_data = json.load(f)

    comp_resume = comp_data.get("resume", [])
    comp_details = comp_data.get("details", {})
    ft_resume = ft_data.get("resume", [])
    ft_details = ft_data.get("details", {})

    cle_ft = ft_resume[0]["modele_approche"] if ft_resume else None
    if cle_ft and cle_ft in comp_details:
        print(f"Les resultats FT sont deja dans la comparaison. Remplacement...")
        for i, r in enumerate(comp_resume):
            if r["modele_approche"] == cle_ft:
                comp_resume[i] = ft_resume[0]
                break
        comp_details[cle_ft] = ft_details[cle_ft]
    else:
        comp_resume.extend(ft_resume)
        comp_details.update(ft_details)

    with open(COMPARAISON_PATH, "w", encoding="utf-8") as f:
        json.dump({"resume": comp_resume, "details": comp_details}, f, indent=2, ensure_ascii=True)

    print(f"Fusion termine. {len(comp_resume)} entrees dans le resume.")

    reponse = input("Regenerer les graphiques ? (o/n): ")
    if reponse.lower() == "o":
        from evaluation.graphiques import generer_graphiques
        generer_graphiques()
        print("Graphiques regeneres dans reports/graphiques/")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/fusionner_resultats_ft.py chemin/vers/resultats_finetuning.json")
        sys.exit(1)
    main(sys.argv[1])

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")

def charger_resultats(path="reports/resultats_comparaison.json"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("resume", [])

def generer_graphiques(resume=None, output_dir="reports/graphiques"):
    if resume is None:
        resume = charger_resultats()
    os.makedirs(output_dir, exist_ok=True)

    modeles = sorted(set(r["modele"] for r in resume))
    approches = sorted(set(r["approche"] for r in resume))
    couleurs = sns.color_palette("husl", len(modeles))
    x = np.arange(len(approches))
    width = 0.8 / len(modeles)

    # 1. F1 par modele x approche
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, modele in enumerate(modeles):
        f1_vals = []
        for approche in approches:
            match = [r for r in resume if r["modele"] == modele and r["approche"] == approche]
            f1_vals.append(match[0]["f1_moyen"] if match else 0)
        offset = (i - len(modeles)/2 + 0.5) * width
        bars = ax.bar(x + offset, f1_vals, width, label=modele, color=couleurs[i], edgecolor="black")
        for bar, val in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_title("Score F1 par Modele et Approche", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(approches)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Modele")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "f1_modele_approche.png"), dpi=150)
    plt.close()
    print(f"Graphique: f1_modele_approche.png")

    # 1b. EM (Exact Match) par modele x approche
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, modele in enumerate(modeles):
        em_vals = []
        for approche in approches:
            match = [r for r in resume if r["modele"] == modele and r["approche"] == approche]
            em_vals.append(match[0]["exact_match_rate"] * 100 if match else 0)
        offset = (i - len(modeles)/2 + 0.5) * width
        bars = ax.bar(x + offset, em_vals, width, label=modele, color=couleurs[i], edgecolor="black")
        for bar, val in zip(bars, em_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", fontsize=8, fontweight="bold")

    ax.set_title("Exact Match (%) par Modele et Approche", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(approches)
    ax.set_ylabel("Exact Match (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="Modele")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "em_modele_approche.png"), dpi=150)
    plt.close()
    print(f"Graphique: em_modele_approche.png")

    # 2. Temps par modele x approche
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, modele in enumerate(modeles):
        temps_vals = []
        for approche in approches:
            match = [r for r in resume if r["modele"] == modele and r["approche"] == approche]
            temps_vals.append(match[0]["temps_moyen"] if match else 0)
        offset = (i - len(modeles)/2 + 0.5) * width
        bars = ax.bar(x + offset, temps_vals, width, label=modele, color=couleurs[i], edgecolor="black")
        for bar, val in zip(bars, temps_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.1f}s", ha="center", fontsize=8, fontweight="bold")

    ax.set_title("Temps de reponse par Modele et Approche", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(approches)
    ax.set_ylabel("Temps (secondes)")
    ax.legend(title="Modele")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "temps_modele_approche.png"), dpi=150)
    plt.close()
    print(f"Graphique: temps_modele_approche.png")

    # 3. Heatmap F1
    fig, ax = plt.subplots(figsize=(10, 6))
    f1_matrix = np.zeros((len(modeles), len(approches)))
    for i, modele in enumerate(modeles):
        for j, approche in enumerate(approches):
            match = [r for r in resume if r["modele"] == modele and r["approche"] == approche]
            f1_matrix[i, j] = match[0]["f1_moyen"] if match else 0

    sns.heatmap(f1_matrix, annot=True, fmt=".2f", xticklabels=approches, yticklabels=modeles,
                cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
    ax.set_title("Heatmap F1: Modeles x Approches", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "heatmap_f1.png"), dpi=150)
    plt.close()
    print(f"Graphique: heatmap_f1.png")

    # 4. Radar pour chaque modele
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    labels = approches
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    for i, modele in enumerate(modeles):
        valeurs = []
        for approche in approches:
            match = [r for r in resume if r["modele"] == modele and r["approche"] == approche]
            valeurs.append(match[0]["f1_moyen"] if match else 0)
        valeurs += valeurs[:1]
        ax.plot(angles, valeurs, "o-", linewidth=2, label=modele, color=couleurs[i])
        ax.fill(angles, valeurs, alpha=0.1, color=couleurs[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
    ax.set_title("Radar F1 par Modele", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "radar_modeles.png"), dpi=150)
    plt.close()
    print(f"Graphique: radar_modeles.png")

    # 5. Tableau resume (image)
    fig, ax = plt.subplots(figsize=(16, len(resume) * 0.5 + 2))
    ax.axis("off")
    cell_text = []
    for r in resume:
        cell_text.append([
            r["modele_approche"],
            f"{r['exact_match_rate']:.0%}",
            f"{r['precision']:.2f}",
            f"{r['recall']:.2f}",
            f"{r['f1_moyen']:.2f}",
            f"{r['temps_moyen']:.1f}s",
        ])
    table = ax.table(cellText=cell_text,
                     colLabels=["Modele / Approche", "EM", "Prec.", "Recall", "F1", "Temps"],
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_fontsize(11)
            cell.set_text_props(fontweight="bold")
    ax.set_title("Tableau comparatif: Modeles x Approches", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "tableau_comparatif.png"), dpi=150)
    plt.close()
    print(f"Graphique: tableau_comparatif.png")

    print(f"\nTous les graphiques sont dans {output_dir}/")

if __name__ == "__main__":
    generer_graphiques()

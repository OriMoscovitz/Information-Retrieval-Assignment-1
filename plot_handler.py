import os
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def plot_iprec_curves(run):
    english_file = os.path.join(os.getcwd(), "outputs",f"run-{run}-trec-en.txt")
    czech_file = os.path.join(os.getcwd(), "outputs",f"run-{run}-trec-cs.txt")

    recall_levels = [i / 10 for i in range(11)]

    def read_iprec_file(filepath):
        iprec = {}

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                metric, _, value = parts[0], parts[1], parts[2]

                match = re.match(r"iprec_at_recall_(\d+\.\d+)", metric)
                if match:
                    recall = float(match.group(1))
                    iprec[recall] = float(value)

        missing = [r for r in recall_levels if r not in iprec]
        if missing:
            raise ValueError(
                f"Missing recall levels in {filepath}: {missing}"
            )

        return [iprec[r] for r in recall_levels]

    english_precision = read_iprec_file(english_file)
    czech_precision = read_iprec_file(czech_file)

    plt.figure(figsize=(8, 5))
    plt.plot(recall_levels, english_precision, marker="o", label="English")
    plt.plot(recall_levels, czech_precision, marker="s", label="Czech")

    plt.xlabel("Recall")
    plt.ylabel("Interpolated Precision")
    plt.title(f"11-point Precision-Recall Curve ({run})")
    plt.xticks(recall_levels)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"11P_AP_results_{run}.png")

    # plt.show()


def plot_run_results(results, lang):
    language = "english" if lang == "en" else "czech"

    steps = sorted(int(run_name.split("-")[1]) for run_name in results)
    map_scores = [results[f"run-{step}"]["map"] for step in steps]
    p10_scores = [results[f"run-{step}"]["P_10"] for step in steps]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, map_scores, marker="o", label="MAP")
    plt.plot(steps, p10_scores, marker="o", label="P_10")
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title(f"{language.capitalize()} Results")
    plt.xticks(steps)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results_{language}.png")
    # plt.show()


def evaluate_and_plot_runs(lang, outputs_dir="outputs", qrels_dir="A1",
                           trec_eval_path="./A1/trec_eval-9.0.7/trec_eval", filename_pattern=None):
    if lang not in {"en", "cs"}:
        raise ValueError("lang must be 'en' or 'cs'")

    outputs_path = Path(outputs_dir)
    qrels_path = Path(qrels_dir) / f"qrels-train_{lang}.txt"
    trec_eval = Path(trec_eval_path)

    if filename_pattern is None:
        pattern = re.compile(rf"^{re.escape(lang)}-(\d+)\.res$")
    else:
        pattern = re.compile(filename_pattern)

    step_to_file = {}
    for file_path in outputs_path.glob("*.res"):
        match = pattern.match(file_path.name)
        if match:
            step_to_file[int(match.group(1))] = file_path

    if not step_to_file:
        raise FileNotFoundError(f"No matching .res files found for {lang}")

    results = {}

    for step in sorted(step_to_file):
        res_file = step_to_file[step]

        cmd = [
            str(trec_eval),
            "-m", "map",
            "-m", "P.10",
            "-M1000",
            str(qrels_path),
            str(res_file),
        ]

        # print(f"command: {cmd}")

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if proc.returncode != 0:
            raise RuntimeError(
                f"trec_eval failed for {res_file.name}\n{proc.stderr}"
            )

        map_score = None
        p10_score = None

        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                metric, value = parts[0], parts[2]
                if metric == "map":
                    map_score = float(value)
                elif metric == "P_10":
                    p10_score = float(value)

        if map_score is None or p10_score is None:
            raise ValueError(f"Could not parse scores for {res_file.name}")

        # print(
        #     f"run-{step} = {map_score:.3f}, {p10_score:.3f}"
        # )

        results[f"run-{step}"] = {
            "map": map_score,
            "P_10": p10_score,
        }

    print(f"--------- Results for {lang} ---------")
    for step in sorted(results.keys()):
        print(f"{step}: map={results[step]["map"]}, P_10={results[step]["P_10"]}")


    plot_run_results(results, lang)

evaluate_and_plot_runs("cs")
evaluate_and_plot_runs("en")

plot_iprec_curves(0)
plot_iprec_curves(1)



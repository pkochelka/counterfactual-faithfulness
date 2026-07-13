import json

import pandas as pd

from measure_agreement import ANNOTATION_DIR, agreement_metrics, parse_scores, print_markdown_table
from sample_for_annotation import JUDGED_DIR, load_judgments


def load_rejudged_faithfulness():
    rows = []
    for (model, dataset, variant, language, eid), record in load_judgments(JUDGED_DIR).items():
        parsed = record.get("parsed", {})
        rows.append({
            "uid": f"{model}__{dataset}_{variant}_{language}__{eid}",
            "new_judge_faithfulness_score": parsed.get("faithfulness_score"),
            "new_judge_incorrect_information": json.dumps(
                parsed.get("incorrect_information", []), ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def write_rejudged_disagreements(annotated, output_path):
    columns = [
        "uid", "Annotator", "prompt_language", "triples", "sentence",
        "human_faithfulness_score", "old_judge_faithfulness_score",
        "new_judge_faithfulness_score", "score_difference",
        "human_faithfulness_reason", "new_judge_incorrect_information",
    ]
    export = annotated.rename(columns={"judge_faithfulness_score": "old_judge_faithfulness_score"}).copy()
    export["human_faithfulness_score"] = export["human_faithfulness_score"].astype(int)
    export["new_judge_faithfulness_score"] = export["new_judge_faithfulness_score"].astype(int)
    export["score_difference"] = (
        export["human_faithfulness_score"] - export["new_judge_faithfulness_score"]
    )
    export = export.sort_values("score_difference", key=abs, ascending=False)
    export[columns].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(export)} rows to {output_path}")


def compare_sample_composition(old_key, new_key):
    shared = set(old_key["uid"]) & set(new_key["uid"])
    print(f"old sample: {len(old_key)} rows, new sample: {len(new_key)} rows, shared uids: {len(shared)}")
    for column in ["faithfulness_stratum", "judge_faithfulness_score", "variant", "size_bucket", "model"]:
        comparison = pd.DataFrame({
            "old_sample": old_key[column].value_counts(),
            "new_sample": new_key[column].value_counts(),
        }).fillna(0).astype(int).sort_index()
        print(f"\n{column}:")
        print(comparison.to_string())


def compare_agreement(annotated):
    for judge_column, label in [
        ("judge_faithfulness_score", "old judge"),
        ("new_judge_faithfulness_score", "new judge"),
    ]:
        report = {"ALL": agreement_metrics(annotated["human_faithfulness_score"], annotated[judge_column])}
        for annotator, group in annotated.groupby("Annotator"):
            report[annotator] = agreement_metrics(group["human_faithfulness_score"], group[judge_column])
        print(f"\n### faithfulness: human vs {label}\n")
        print_markdown_table(pd.DataFrame(report).T.round(3))


def main():
    old_key = pd.read_csv(ANNOTATION_DIR / "annotation_key_old.csv", encoding="utf-8-sig")
    new_key = pd.read_csv(ANNOTATION_DIR / "annotation_key.csv", encoding="utf-8-sig")
    compare_sample_composition(old_key, new_key)

    annotated = pd.read_csv(ANNOTATION_DIR / "annotation_sample_old.csv", encoding="utf-8-sig")
    annotated["human_faithfulness_score"] = parse_scores(annotated["human_faithfulness_score"])
    annotated = annotated.merge(
        old_key[["uid", "judge_faithfulness_score"]], on="uid", how="inner", validate="one_to_one"
    )
    annotated = annotated.merge(load_rejudged_faithfulness(), on="uid", how="left", validate="one_to_one")
    annotated = annotated.dropna(subset=[
        "human_faithfulness_score", "judge_faithfulness_score", "new_judge_faithfulness_score"
    ])
    print(f"\nhuman-annotated rows with both old and new judge scores: {len(annotated)}")

    write_rejudged_disagreements(annotated, ANNOTATION_DIR / "faithfulness_disagreements_rejudged.csv")

    judge_shift = (annotated["new_judge_faithfulness_score"] - annotated["judge_faithfulness_score"]).astype(int)
    print("\nnew judge minus old judge, per human-annotated row:")
    print(judge_shift.value_counts().sort_index().to_string())

    compare_agreement(annotated)


if __name__ == "__main__":
    main()
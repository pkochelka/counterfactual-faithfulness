import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from generate_speeches import call_llm
from pathlib import Path
from typing import Literal
import pandas as pd
from tqdm import tqdm

from triplet_dataclasses import PhrasedTriple, Triple, MemberType, TripleMember
from constants import FACTUAL_PHRASED_DATA, CUS_QA_TEXT_CZ
from dataclasses import dataclass
from xml_tagged import export_to_xml_file


def load_prompt(name: Literal["prompt", "prompt-tag-czech"]) -> str:
    prompt_file = Path(__file__).parent / f"{name}.txt"
    with open(prompt_file, "r", encoding="utf8") as f:
        prompt = f.read()

    return prompt


def fill_prompt(prompt: str, qa: "DataPoint") -> str:
    return prompt + f"\nQ: {qa.question}\nA: {qa.answer}"


def fill_tag_prompt(prompt: str, phrase: str, words: list[str]) -> str:
    return prompt + f"\nVěta: {phrase}\nSlova: {', '.join(words)}"


@dataclass
class DataPoint:
    id_: int
    category: str
    question: str
    answer: str


if __name__ == "__main__":
    df = pd.read_parquet(CUS_QA_TEXT_CZ, engine="pyarrow")

    datapoints = [
        DataPoint(
            id_=row.id,
            category=row.category,
            question=row.question_orig,
            answer=row.answer_orig,
        )
        for row in df[["id", "category", "question_orig", "answer_orig"]].itertuples(
            index=False
        )
    ]

    prompt_base = load_prompt("prompt")

    def parse_phrased_triples(text: str, data: DataPoint) -> PhrasedTriple:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

        phrase = lines[1].removeprefix("<phrase>").removesuffix("</phrase>")
        triples = []
        for line in lines[1:]:
            if "|" in line:
                line = line.removeprefix("<mtriple>").removesuffix("</mtriple>")
                subject, predicate, object_ = [p.strip() for p in line.split("|")]
                triples.append(
                    Triple(TripleMember(subject), predicate, TripleMember(object_))
                )

        return PhrasedTriple(
            id_=data.id_, category=data.category, phrase=phrase, triples=triples
        )

    tag_prompt = load_prompt("prompt-tag-czech").format(
        member_types=", ".join(str(t) for t in MemberType)
    )

    pts: list[PhrasedTriple] = []

    for datapoint in tqdm(datapoints):
        response = call_llm(fill_prompt(prompt_base, datapoint), "mistral-medium-3.5")
        pt = parse_phrased_triples(response, datapoint)

        assert pt.phrase is not None

        tag_response = call_llm(
            fill_tag_prompt(tag_prompt, pt.phrase, pt.get_unique_members()),
            "mistral-medium-3.5",
        )
        pt._tag_members(tag_response)
        print()
        print(pt)

        pts.append(pt)

    export_to_xml_file(pts, FACTUAL_PHRASED_DATA)

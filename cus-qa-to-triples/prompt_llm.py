import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from generate_speeches import call_llm
from pathlib import Path
from typing import Literal
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

from triplet_dataclasses import PhrasedTriple, Triple, CZMemberType, TripleMember
from constants import (
    CUS_QA_TEXT_SK,
    CUS_QA_TEXT_CZ,
    SK_DATA,
    CZ_DATA,
    CZMemberType,
    SKMemberType
)
from dataclasses import dataclass
from xml_tagged import export_to_xml_file


def load_prompt(name: Literal["prompt", "prompt-tag"], language: Literal["cz", "sk"]) -> str:
    if name == "prompt-tag":
        name = f"{name}-{language}" # type: ignore
    prompt_file = Path(__file__).parent / f"{name}.txt"
    with open(prompt_file, "r", encoding="utf8") as f:
        prompt = f.read()

    return prompt


def fill_prompt(prompt: str, qa: "DataPoint", language: Literal["cz", "sk"]) -> str:
    if language == "cz":
        lang_name = "Czech"
    elif language == "sk":
        lang_name = "Slovak"
    else:
        raise ValueError()
    
    prompt.format(language=lang_name)
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
    parser = ArgumentParser()
    parser.add_argument("language", choices=["cz", "sk"])
    args = parser.parse_args()

    if args.language == "cz":
        args.dataset = CUS_QA_TEXT_CZ
        args.output = CZ_DATA / "fa-phrase.xml"
        args.member_type = CZMemberType
    elif args.language == "sk":
        args.dataset = CUS_QA_TEXT_SK
        args.output = SK_DATA / "fa-phrase.xml"
        args.member_type = SKMemberType
    else:
        raise ValueError()
    
    df = pd.read_parquet(args.dataset, engine="pyarrow")

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

    prompt_base = load_prompt("prompt", args.language)

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

    tag_prompt = load_prompt("prompt-tag", args.language).format(
        member_types=", ".join(str(t) for t in args.member_type)
    )

    pts: list[PhrasedTriple] = []

    for datapoint in tqdm(datapoints):
        response = call_llm(fill_prompt(prompt_base, datapoint, args.language), "mistral-medium-3.5")
        pt = parse_phrased_triples(response, datapoint)

        assert pt.phrase is not None

        tag_response = call_llm(
            fill_tag_prompt(tag_prompt, pt.phrase, pt.get_unique_members()),
            "mistral-medium-3.5",
        )
        pt._tag_members(tag_response, args.member_type)
        print()
        print(pt)

        pts.append(pt)

    export_to_xml_file(pts, args.output)

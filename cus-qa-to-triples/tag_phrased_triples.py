from argparse import ArgumentParser
from xml_tagged import export_to_xml_file, import_from_xml_file
from constants import CZ_DATA, SK_DATA, _MemberType, CZMemberType, SKMemberType
from typing import Literal
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from generate_speeches import call_llm


def _retrieve_tags(tags: str, language: Literal["cz", "sk"]) -> set[_MemberType]:
    type_ = SKMemberType if language == "sk" else CZMemberType
    output = set()
    for t in tags.split(","):
        output.add(type_(t))
    return output


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("language", type=str, choices=["cz", "sk"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-changes",
        type=int,
        default=2,
        help="Number of maximum changes made to a phrase. Randomly chosen between 1 and 'max_changes'",
    )
    parser.add_argument(
        "--tags",
        type=str,
        help=f"Tags that will be changed separated by comma. Default is all. {[t.value for t in CZMemberType]}",
    )

    args = parser.parse_args()
    if args.tags is not None:
        args.tags = _retrieve_tags(args.tags, args.language)

    if args.language == "cz":
        dir = CZ_DATA
        types = CZMemberType
        wiki_lang = "cs"
    elif args.language == "sk":
        dir = SK_DATA
        types = SKMemberType
        wiki_lang = "sk"
    else:
        raise ValueError()

    factual_data_file = dir / "fa-phrase.xml"

    pts = import_from_xml_file(factual_data_file, args.language)

    loaded_count = len(pts)

    def load_prompt(language: Literal["cz", "sk"]) -> str:
        name = "prompt-tag-afterwards"
        name = f"{name}-{language}"  # type: ignore
        prompt_file = Path(__file__).parent / f"{name}.txt"
        with open(prompt_file, "r", encoding="utf8") as f:
            prompt = f.read()

        return prompt

    prompt = load_prompt(args.language)

    def normalize_members(members: list[str]) -> list[str]:
        return [m.replace("_", " ") for m in members]

    all_keywords = set()
    for pt in pts:
        all_keywords.update(normalize_members(pt.get_unique_members()))

    # all_keywords = list(all_keywords)
    # with open(f"wiki-data-{args.language}.txt", "w", encoding="utf8") as file:
    #     for i in tqdm(range(0, len(all_keywords), BATCH_SIZE)):
    #         batch = all_keywords[i: i + BATCH_SIZE]
    #         for key, value in fetch_batch(batch, wiki_lang).items():
    #             file.write(f"{key}==={value}\n")
    
    keywords: dict[str, str] = dict()
    with open(f"wiki-data-{args.language}.txt", "r", encoding="utf8") as file:
        for line in file:
            line = line.strip()
            keyword, wiki = line.split("===")
            keyword = keyword.replace("_", " ")
            keywords[keyword] = wiki
    
    DEBUG = False
    for pt in tqdm(pts):
        if DEBUG: print(pt.phrase)
        for m in pt.get_unique_members():
            wiki_context = keywords.get(m, "none")
            filled_prompt = prompt.format(
                phrase=pt.phrase,
                keyword=m,
                wiki_context=wiki_context
            )

            result = call_llm(filled_prompt, "gpt-oss-120b")
            if DEBUG: print(f"{m:<20} = {result}")
            type_ = types.from_str(result)
            pt.update_member_tags(m, type_)

        if DEBUG: print("---------------")
    
    export_to_xml_file(pts, factual_data_file.with_name(factual_data_file.name + "-new"))
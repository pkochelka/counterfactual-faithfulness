from create_CF_data import change_facts_single, _retrieve_tags
from pathlib import Path
import json
from copy import copy
import random
from collections import defaultdict
from argparse import ArgumentParser
from xml_tagged import export_to_xml_file, import_from_xml_file
from triplet_dataclasses import PhrasedTriple, TripleMember
from constants import CZ_DATA, SK_DATA, _MemberType, CZMemberType, SKMemberType
from typing import Optional, Literal


def load_fictional_assets_to_triple_members(
    language: Literal["cz", "sk"],
) -> defaultdict[_MemberType, list[TripleMember]]:
    assert language in {"cz", "sk"}
    file = (
        Path(__file__).parent
        / "data"
        / "fictional-assets"
        / f"{language}-fictional-entities.json"
    )
    if language == "cz":
        type_ = CZMemberType
    elif language == "sk":
        type_ = SKMemberType
    else:
        raise ValueError

    with open(file, "r", encoding="utf8") as f:
        data = json.load(f)

    output: defaultdict[_MemberType, list[TripleMember]] = defaultdict(list)

    for t in type_:
        output[t] = []
        for entity in data[t]:
            output[t].append(TripleMember(entity, t))
    return output


def make_fictional_single(
    pt: PhrasedTriple,
    options: defaultdict[_MemberType, list[TripleMember]],
    max_changes: int,
    rng: random.Random,
) -> PhrasedTriple | None:
    options_filtered = defaultdict(list)
    members = set(pt.get_unique_members())
    # for key, value in options.items():
    #     options_filtered[key] = [o for o in value if o not in members]

    return change_facts_single(pt, copy(options), max_changes, rng)


def create_FI_for_single_language(
    language: Literal["cz", "sk"],
    max_changes: int = 2,
    tags_to_change: Optional[set[_MemberType]] = None,
    seed: int = 42,
) -> None:

    if language == "cz":
        dir = CZ_DATA
    elif language == "sk":
        dir = SK_DATA
    else:
        raise ValueError()

    fa_data_file = dir / "fa-phrase.xml"
    fi_data_output = dir / "fi-phrase.xml"

    options = load_fictional_assets_to_triple_members(language)
    print({key: len(value) for key, value in options.items()})
    
    options_to_keep = defaultdict(list)
    if tags_to_change is not None:
        for tag in tags_to_change:
            options_to_keep[tag] = options[tag]
        options = options_to_keep
    
    
    pts = import_from_xml_file(fa_data_file, language)

    loaded_count = len(pts)

    new_pts: list[PhrasedTriple] = []

    rng = random.Random(seed)

    for pt in pts:
        fi_pt = make_fictional_single(pt, options, max_changes, rng)
        if fi_pt is not None:
            new_pts.append(fi_pt)

    print(
        f"Created {len(new_pts)} Fictional triples from {loaded_count} Factual triples"
    )

    export_to_xml_file(new_pts, fi_data_output)


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

    create_FI_for_single_language(
        args.language, args.max_changes, tags_to_change=args.tags, seed=args.seed
    )

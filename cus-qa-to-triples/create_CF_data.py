import random
from collections import defaultdict
from argparse import ArgumentParser
from xml_tagged import export_to_xml_file, import_from_xml_file
from triplet_dataclasses import PhrasedTriple, Triple, MemberType, TripleMember
from constants import FACTUAL_PHRASED_DATA, COUNTER_FACTUAL_PHRASED_DATA
from typing import Optional


def change_facts_single(
        pt: PhrasedTriple,
        options: defaultdict[MemberType, list[TripleMember]],
        max_changes: int,
        rng: random.Random,
) -> PhrasedTriple | None:
    new_pt = PhrasedTriple(pt.id_, pt.category, None, [])

    to_replace: set[TripleMember] = set(pt.get_unique_tagged_members())

    replace_with: dict[TripleMember, TripleMember] = dict()

    for m in to_replace:
        assert m.type_ is not None
        m_options = options[m.type_]

        # skip if there are no options
        if len(m_options) == 0:
            continue
        # or all options are equal
        if all(m == pm for pm in m_options):
            continue

        new_m = None
        # this could potentially be infinite, but meh
        while new_m is None:
            chosen_m = rng.choice(m_options)
            if chosen_m != m:
                new_m = chosen_m

        replace_with[m] = new_m

    
    # unchanged
    if len(replace_with) == 0:
        return None
    
    def _shuffle_dict_and_choose(d: dict, first_n: int, rng: random.Random) -> dict:
        """
        Shuffles dict and chooses the first n
        changes that will be applied.
        """
        lst = list(d.items())[:first_n]
        rng.shuffle(lst)
        return dict(lst)

    changes = rng.randint(1, max_changes)
    
    replace_with = _shuffle_dict_and_choose(replace_with, changes, rng)
    
    for from_, to in replace_with.items():
        print(pt.id_)
        print(f"- Replacing {from_} for {to}")

    for triplet in pt.triples:
        new_o = replace_with.get(triplet.object_, triplet.object_)
        new_s = replace_with.get(triplet.subject, triplet.subject)
        
        new_pt.triples.append(
            Triple(
                subject=new_s,
                predicate=triplet.predicate,
                object_=new_o
            )
        )

    return new_pt


def change_facts_multiple(
        pts: list[PhrasedTriple],
        max_changes: int = 2,
        tags_to_change: Optional[set[MemberType]] = None,
        seed: int = 42
) -> list[PhrasedTriple]:
    assert max_changes > 0

    rng = random.Random(seed)

    options: defaultdict[MemberType, list[TripleMember]] = defaultdict(list)

    for pt in pts:
        for m in pt.get_unique_tagged_members():
            assert m.type_ is not None
            if tags_to_change is None or m.type_ in tags_to_change:
                options[m.type_].append(m)

    output: list[PhrasedTriple] = []
    for pt in pts:
        new_pt = change_facts_single(pt, options, max_changes, rng)
        if new_pt is not None:
            output.append(new_pt)
        
    return output


def _retrieve_tags(tags: str) -> set[MemberType]:
    output = set()
    for t in tags.split(","):
        output.add(MemberType(t))
    return output


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-changes", type=int, default=2,
        help="Number of maximum changes made to a phrase. Randomly chosen between 1 and 'max_changes'"
    )
    parser.add_argument(
        "--tags", type=str,
        help=f"Tags that will be changed separated by comma. Default is all. {[t.value for t in MemberType]}"
    )

    args = parser.parse_args()
    if args.tags is not None:
        args.tags = _retrieve_tags(args.tags)

    pts = import_from_xml_file(FACTUAL_PHRASED_DATA)
    
    pts = change_facts_multiple(
        pts,
        max_changes=args.max_changes,
        tags_to_change=args.tags,
        seed=args.seed
    )

    export_to_xml_file(pts, COUNTER_FACTUAL_PHRASED_DATA)

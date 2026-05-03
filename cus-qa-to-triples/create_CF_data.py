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
    to_replace: list[TripleMember] = sorted(
        set(pt.get_unique_tagged_members()),
        key=lambda m: (m.type_.value, m.value),
    )
    replace_with: dict[TripleMember, TripleMember] = dict()

    for m in to_replace:
        assert m.type_ is not None
        # exclude members belonging to this entry
        m_options = [o for o in options[m.type_] if o != m]

        if len(m_options) == 0:
            continue

        replace_with[m] = rng.choice(
            m_options
        )  # guaranteed different, no while loop needed

    if len(replace_with) == 0:
        return None

    def _shuffle_dict_and_choose(
        d: dict[TripleMember, list[TripleMember]], first_n: int, rng: random.Random
    ) -> dict:
        lst = list(d.items())
        # def _normalize(record: tuple[TripleMember, list[TripleMember]]) -> str:
        #     return record[0]._normalized + "".join(n._normalized for n in sorted(record[1], key=lambda t: t._normalized))
        # lst.sort(key=lambda t: _normalize(t))
        rng.shuffle(lst)
        return dict(lst[:first_n])

    changes = rng.randint(1, max_changes)

    replace_with = _shuffle_dict_and_choose(replace_with, changes, rng)

    if len(replace_with) == 0:
        return None

    for from_, to in replace_with.items():
        print(pt.id_)
        print(f"- Replacing {from_} for {to}")

    changed = False
    for triplet in pt.triples:
        new_o = replace_with.get(triplet.object_)
        new_s = replace_with.get(triplet.subject)

        if new_o is not None:
            changed = True
        else:
            new_o = triplet.object_
        if new_s is not None:
            changed = True
        else:
            new_s = triplet.subject

        new_pt.triples.append(
            Triple(subject=new_s, predicate=triplet.predicate, object_=new_o)
        )

    if not changed:
        return None

    return new_pt


def change_facts_multiple(
    pts: list[PhrasedTriple],
    max_changes: int = 2,
    tags_to_change: Optional[set[MemberType]] = None,
    seed: int = 42,
) -> list[PhrasedTriple]:
    assert max_changes > 0

    rng = random.Random(seed)

    options: defaultdict[MemberType, list[TripleMember]] = defaultdict(list)

    for pt in sorted(pts, key=lambda p: p.id_):
        for m in sorted(pt.get_unique_tagged_members(), key=lambda m: (m.type_.value, m.value)):
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
        "--max-changes",
        type=int,
        default=2,
        help="Number of maximum changes made to a phrase. Randomly chosen between 1 and 'max_changes'",
    )
    parser.add_argument(
        "--tags",
        type=str,
        help=f"Tags that will be changed separated by comma. Default is all. {[t.value for t in MemberType]}",
    )

    args = parser.parse_args()
    if args.tags is not None:
        args.tags = _retrieve_tags(args.tags)

    pts = import_from_xml_file(FACTUAL_PHRASED_DATA)

    loaded_count = len(pts)

    pts = change_facts_multiple(
        pts, max_changes=args.max_changes, tags_to_change=args.tags, seed=args.seed
    )

    print(f"Created {len(pts)} CF triples from {loaded_count} F triples")
    export_to_xml_file(pts, COUNTER_FACTUAL_PHRASED_DATA)

from xml.etree.ElementTree import Element, SubElement
from pathlib import Path

from xml_tagged import import_from_xml_file
from triplet_dataclasses import PhrasedTriple
from xml_pretty import root_to_pretty_xml


def tagged_to_triples_xml(pts: list[PhrasedTriple]) -> Element:
    root = Element("benchmark")
    entries = SubElement(root, "entries")

    for pt in pts:
        if len(pt) == 0:
            print(f"{pt.id_} does not contain any triples")
            continue

        entry = Element(
            "entry",
            {
                "category": str(pt.category),
                "eid": f"Id{pt.id_}",
                "shape": "NA",
                "shape_type": "NA",
                "size": str(len(pt)),
            },
        )

        triples = SubElement(entry, "originaltripleset")
        for t in pt.triples:
            SubElement(triples, "otriple").text = t.str_without_tags()

        entries.append(entry)

    return root


def tagged_file_to_triples_xml_file(source: Path, output: Path) -> None:
    pts = import_from_xml_file(source)
    root = tagged_to_triples_xml(pts)
    str_xml = root_to_pretty_xml(root)
    with open(output, "w", encoding="utf8") as f:
        f.write(str_xml)

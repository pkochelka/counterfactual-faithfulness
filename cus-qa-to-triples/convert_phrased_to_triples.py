from pathlib import Path
from typing import Literal

from xml_triples import import_from_xml_file
from xml_pretty import root_to_pretty_xml
from xml_triples import tagged_to_triples_xml


def convert_phrased_to_triples(source: Path, output: Path, lang: Literal["cz", "sk"]) -> None:
    pts = import_from_xml_file(source, lang)
    root = tagged_to_triples_xml(pts)
    str_xml = root_to_pretty_xml(root)
    with open(output, "w", encoding="utf8") as f:
        f.write(str_xml)
        
    print(f"Converted phrased triples '{source}' to triples '{output}'")

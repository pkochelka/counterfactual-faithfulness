from constants import (
    SK_DATA,
    CZ_DATA
)
from xml_triples import tagged_file_to_triples_xml_file

if __name__ == "__main__":
    for dir, lang in [(CZ_DATA, "cz"), (SK_DATA, "sk")]:
        tagged_file_to_triples_xml_file(
            dir / "cf-phrase.xml",
            dir / "cf.xml",
            lang
        )
        tagged_file_to_triples_xml_file(
            dir / "fa-phrase.xml",
            dir / "fa.xml",
            lang
        )
    
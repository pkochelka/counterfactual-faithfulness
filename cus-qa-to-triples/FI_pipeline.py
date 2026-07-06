from pathlib import Path

from clean_up_raw_FI_data import main as clean_up_main
from create_FI_data import create_FI_for_single_language
from constants import CZMemberType, SKMemberType
from convert_phrased_to_triples import convert_phrased_to_triples


def main() -> None:
    clean_up_main()

    create_FI_for_single_language(
        language="cz",
        max_changes=1,
        tags_to_change={
            CZMemberType.MISTO,
            CZMemberType.JMENO_CLOVEK,
            CZMemberType.CISLO,
            CZMemberType.DATUM,
        },
    )

    create_FI_for_single_language(
        language="sk",
        max_changes=1,
        tags_to_change={
            SKMemberType.MIESTO,
            SKMemberType.MENO_CLOVEK,
            SKMemberType.CISLO,
            SKMemberType.DATUM,
        },
    )

    for lang in ["cz", "sk"]:
        source = Path(__file__).parent / "data" / lang / "fi-phrase.xml"
        output = Path(__file__).parent / "data" / lang / "fi.xml"
        convert_phrased_to_triples(source, output, lang)


if __name__ == "__main__":
    main()

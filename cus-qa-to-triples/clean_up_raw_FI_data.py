from typing import Literal
import json
from pathlib import Path

from constants import CZMemberType, SKMemberType

from datetime import datetime


def is_valid_date(date_str: str, fmt: str = "%d-%m-%Y") -> bool:
    try:
        datetime.strptime(date_str, fmt)
        return True
    except ValueError:
        return False


def is_valid_number(number_str: str) -> bool:
    try:
        int(number_str)
        return True
    except ValueError:
        return False


def normalize_string(text: str, lower: bool = False) -> str:
    text = text.replace(" ", "_")
    if lower:
        return text.lower()
    return text


def _clean_data(data: dict[str, list[str]]) -> dict[str, list[str]]:
    # names and places
    for name_type in [
        CZMemberType.JMENO_UMELECKE_DILO,
        SKMemberType.MENO_UMELECKE_DIELO,
        CZMemberType.JMENO_CLOVEK,
        SKMemberType.MENO_CLOVEK,
        CZMemberType.MISTO,
        SKMemberType.MENO,
        CZMemberType.MISTO,
        SKMemberType.MIESTO,
    ]:
        names = data.get(name_type)
        if names is not None:
            names = [normalize_string(n) for n in set(names)]
            names.sort()
            data[name_type] = names

    # dates
    dates = data[CZMemberType.DATUM]
    dates = [d for d in dates if is_valid_date(d)]
    data[CZMemberType.DATUM] = sorted(set(dates))

    # numbers
    numbers = data[CZMemberType.CISLO]
    numbers = [n.replace(" ", "") for n in numbers]
    numbers = [n for n in numbers if is_valid_number(n)]
    data[CZMemberType.CISLO] = sorted(set(numbers))

    # occupation
    for name_type in [
        CZMemberType.ZAMESTNANI,
        SKMemberType.ZAMESTNANIE,
    ]:
        names = data.get(name_type)
        if names is not None:
            names = [normalize_string(n, True) for n in set(names)]
            names.sort()
            data[name_type] = names

    return data


CZ_FICTIONAL_DATA = Path(__file__).parent / "data" / "fictional-assets" / "cz-raw.json"
SK_FICTIONAL_DATA = Path(__file__).parent / "data" / "fictional-assets" / "sk-raw.json"


def clean_file(language: Literal["cz", "sk"]) -> None:
    if language == "cz":
        file = CZ_FICTIONAL_DATA
    elif language == "sk":
        file = SK_FICTIONAL_DATA
    else:
        raise ValueError

    with open(file, "r", encoding="utf8") as f:
        data = json.load(f)

    data = _clean_data(data)

    output = file.with_name(f"{language}-fictional-entities.json")

    with open(output, "w", encoding="utf8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Cleaned '{language}', saved to '{output}")


def main() -> None:
    for language in ["cz", "sk"]:
        clean_file(language)


if __name__ == "__main__":
    main()

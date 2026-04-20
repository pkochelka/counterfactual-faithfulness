import xml.etree.ElementTree as ET
import pandas as pd


def parse_triple(text: str) -> tuple[str, str, str]:
    """Parse 'Subject | Predicate | Object' into a tuple."""
    parts = [p.strip() for p in text.split("|", 2)]
    if len(parts) != 3:
        raise ValueError(f"Malformed triple: {text!r}")
    return parts[0], parts[1], parts[2]


def load_xml(path: str) -> pd.DataFrame:
    """Parse a benchmark XML file into a long-format DataFrame."""
    tree = ET.parse(path)
    root = tree.getroot()

    rows = []
    for entry in root.findall("entries/entry"):
        meta = {
            "eid": entry.get("eid"),
            "category": entry.get("category"),
            "shape": entry.get("shape"),
            "shape_type": entry.get("shape_type"),
            "size": int(entry.get("size", 0)),
        }

        for kind, xpath in [("original", "originaltripleset/otriple"),
                            ("modified", "modifiedtripleset/mtriple")]:
            for triple_elem in entry.findall(xpath):
                if triple_elem.text is None:
                    continue
                subject, predicate, obj = parse_triple(triple_elem.text)
                rows.append({
                    **meta,
                    "kind": kind,
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                })

    df = pd.DataFrame(rows)

    for col in ("category", "shape", "shape_type", "kind", "predicate"):
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


if __name__ == "__main__":
    df = load_xml("./data/GEM-v2-D2T-SharedTask/D2T-1-FI_WebNLG_Fictional.xml")

    print(f"Loaded {len(df):,} triples from {df['eid'].nunique():,} entries")
    print(df.head(10))
    print()
    print("Triples per kind:")
    print(df["kind"].value_counts())
    print()
    print("Top 5 categories by entry count:")
    print(df.groupby("category", observed=True)["eid"].nunique().nlargest(5))
    df.to_csv('data/webnlg_fi.csv', index=False)
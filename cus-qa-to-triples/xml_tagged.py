from xml.etree.ElementTree import Element, SubElement, ElementTree, indent, parse
from io import StringIO
from pathlib import Path
from typing import Literal

from triplet_dataclasses import PhrasedTriple, Triple, TripleMember
from constants import CZMemberType, SKMemberType
from xml_pretty import root_to_pretty_xml

# Export

def member_to_xml(parent: Element, tag: str, member: TripleMember) -> Element:
    el = SubElement(parent, tag)
    el.set("value", member.value)
    el.set("type", member.type_.value if member.type_ is not None else "null")
    return el

def triple_to_xml(parent: Element, triple: Triple) -> Element:
    el = SubElement(parent, "triple")
    member_to_xml(el, "subject", triple.subject)
    pred = SubElement(el, "predicate")
    pred.text = triple.predicate
    member_to_xml(el, "object", triple.object_)
    return el

def phrased_triple_to_xml(pt: PhrasedTriple, parent: Element) -> Element:
    root = SubElement(parent, "phrased_triple")
    root.set("id", str(pt.id_))
    root.set("category", pt.category)
    phrase_el = SubElement(root, "phrase")
    phrase_el.text = pt.phrase or ""
    triples_el = SubElement(root, "triples")
    for triple in pt.triples:
        triple_to_xml(triples_el, triple)
    return root

def export_to_xml_string(pts: list[PhrasedTriple]) -> str:
    root = Element("phrased_triples")
    for pt in pts:
        phrased_triple_to_xml(pt, root)
    indent(root, space="    ")
    buf = StringIO()
    ElementTree(root).write(buf, encoding="unicode", xml_declaration=False)
    return buf.getvalue()

def export_to_xml_file(pts: list[PhrasedTriple], path: str | Path) -> None:
    root = Element("phrased_triples")
    for pt in pts:
        phrased_triple_to_xml(pt, root)
    
    
    with open(path, "w", encoding="utf8") as f:
        f.write(root_to_pretty_xml(root))
        
# Import

def member_from_xml(el: Element, language: Literal["cz", "sk"]) -> TripleMember:
    tp = CZMemberType if language == "cz" else SKMemberType
    value = el.get("value", "")
    raw_type = el.get("type", "null")
    try:
        type_ = tp(raw_type)
    except ValueError:
        type_ = None
    member = TripleMember(value=value)
    member.type_ = type_
    return member

def phrased_triple_from_xml(root: Element, language: Literal["cz", "sk"]) -> PhrasedTriple:
    id_ = int(root.get("id", 0))
    category = root.get("category", "")
    phrase = root.findtext("phrase") or None
    pt = PhrasedTriple(id_=id_, category=category, phrase=phrase)
    for triple_el in root.findall("triples/triple"):
        subj_el = triple_el.find("subject")
        pred_el = triple_el.find("predicate")
        obj_el  = triple_el.find("object")
        if subj_el is None or pred_el is None or obj_el is None:
            continue
        pt.triples.append(Triple(
            subject   = member_from_xml(subj_el, language),
            predicate = pred_el.text or "",
            object_   = member_from_xml(obj_el, language),
        ))
    return pt

def import_from_xml_string(xml: str, language: Literal["cz", "sk"]) -> list[PhrasedTriple]:
    root = parse(StringIO(xml)).getroot()
    return [phrased_triple_from_xml(el, language) for el in root.findall("phrased_triple")]

def import_from_xml_file(path: str | Path, language: Literal["cz", "sk"]) -> list[PhrasedTriple]:
    root = parse(path).getroot()
    return [phrased_triple_from_xml(el, language) for el in root.findall("phrased_triple")]
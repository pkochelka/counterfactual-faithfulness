from dataclasses import dataclass, field
from typing import Optional, Any

from constants import MemberType


import unicodedata

def strip_accents(s):
    # Source - https://stackoverflow.com/a/518232
    # Posted by oefe, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-05-02, License - CC BY-SA 3.0
   return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_str_for_eq(value: str) -> str:
    """
    Removes white space, underscores, accents
    and makes value lowercase.
    """
    return strip_accents(value.replace("_", ""))


@dataclass
class TripleMember:
    value: str
    type_: Optional[MemberType] = None

    _normalized: str = field(init=False, repr=False) 

    def __post_init__(self) -> None:
        self.value = self.value.replace(" ", "_")
        self._normalized = normalize_str_for_eq(self.value)
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, TripleMember):
            other = other._normalized
        
        return other == self._normalized
    
    def __str__(self) -> str:
        return f"{self.value} [{self.type_}]"
    
    def __hash__(self) -> int:
        return hash(self._normalized)

@dataclass
class Triple:
    subject: TripleMember
    predicate: str
    object_: TripleMember

    def __str__(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.object_}"
    
    def str_without_tags(self) -> str:
        return f"{self.subject.value} | {self.predicate} | {self.object_.value}"


@dataclass
class PhrasedTriple:
    id_: int
    category: str
    phrase: Optional[str]
    triples: list[Triple] = field(default_factory=list)

    def __str__(self) -> str:
        _phrase = self.phrase if self.phrase is not None else "no phrase"
        return _phrase + "\n" + "\n triplet: ".join(str(t) for t in self.triples)

    def __len__(self) -> int:
        return len(self.triples)
    
    def get_unique_members(self) -> list[str]:
        output = set()
        for triplet in self.triples:
            output.add(triplet.subject.value)
            output.add(triplet.object_.value)
        
        return list(output)
    
    
    def get_unique_tagged_members(self) -> list[TripleMember]:
        output: set[TripleMember] = set()
        for triplet in self.triples:
            for ject in [triplet.subject, triplet.object_]:
                if ject.type_ is not None:
                    output.add(ject)
                    output.add(ject)
        
        return list(output)    
    
    def _tag_members(self, llm_output: str) -> None:
        mapping: dict[str, MemberType] = dict()

        for line in llm_output.split("\n"):
            try:
                word, tag = line.split(":")
                word = word.strip()
                tag = tag.strip()

                word = normalize_str_for_eq(word)
                tag = MemberType(tag)

                mapping[word] = tag
            except Exception as e:
                print(e, line)
        
        for t in self.triples:
            def _tag(mapping: dict[str, MemberType], member: TripleMember) -> None:
                type_ = mapping.get(member._normalized)
                if type_ is not None:
                    member.type_ = type_
            
            for m in [t.object_, t.subject]:
                _tag(mapping, m)


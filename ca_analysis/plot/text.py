from typing import (
    List,
    Mapping,
    Optional,
    Sequence,    
)

from ..resampling import resample_labels


__all__ = [
    "simplify_labels",    
]



def simplify_labels(
    labels: Sequence[str],
    replace: Optional[Mapping[str, str]] = None,
    ) -> List[str]:
    
    out = []
    for i, elt in enumerate(labels):
        if elt.endswith(".front") or elt.endswith(".back"):
            elt = ""
        elt = elt.split("_")[0]
        out.append(elt)

    labels = out
    if replace:
        for i, elt in enumerate(labels):
            if elt in replace:
                labels[i] = replace[elt]
    return labels


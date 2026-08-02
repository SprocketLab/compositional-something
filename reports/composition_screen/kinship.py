"""Kinship algebra for CLUTRR, as paths over parent/child/spouse steps.

R(x, y) reads "y is x's R", so each relation is a walk from x to y:
  father P | mother P | son C | daughter C | husband S | wife S
  brother PC | sister PC        (up to a parent, back down to a child)
  grandfather PP | grandmother PP | grandson CC | granddaughter CC
  uncle PPC | aunt PPC          (parent's sibling)
  nephew PCC | niece PCC        (sibling's child)
  father-in-law SP | mother-in-law SP | son-in-law CS | daughter-in-law CS

Composition concatenates the walks and cancels CP (a child's parent is
oneself) and SS (a spouse's spouse is oneself).  PC does NOT cancel -- that is
precisely the sibling relation.  Mining the rule from data instead yields only
the 62 pairs the CLUTRR generator happens to traverse, which is why arbitrary
split points produced "illegal" compositions like nephew-brother.
"""

PATHS = {
    "father": ("P", "m"), "mother": ("P", "f"),
    "son": ("C", "m"), "daughter": ("C", "f"),
    "husband": ("S", "m"), "wife": ("S", "f"),
    "brother": ("PC", "m"), "sister": ("PC", "f"),
    "grandfather": ("PP", "m"), "grandmother": ("PP", "f"),
    "grandson": ("CC", "m"), "granddaughter": ("CC", "f"),
    "uncle": ("PPC", "m"), "aunt": ("PPC", "f"),
    "nephew": ("PCC", "m"), "niece": ("PCC", "f"),
    "father-in-law": ("SP", "m"), "mother-in-law": ("SP", "f"),
    "son-in-law": ("CS", "m"), "daughter-in-law": ("CS", "f"),
}
BY_PATH = {(p, g): name for name, (p, g) in PATHS.items()}


# CP: a child's parent is oneself.  SS: a spouse's spouse is oneself.
# SC -> C: CLUTRR has no step-children, so a spouse's child is one's own child.
# Deliberately asymmetric -- SP does NOT reduce, because a spouse's parent is an
# in-law rather than a parent.
REDUCTIONS = (("CP", ""), ("SS", ""), ("SC", "C"))


def reduce_path(path: str) -> str:
    changed = True
    while changed:
        changed = False
        for src, dst in REDUCTIONS:
            if src in path:
                path = path.replace(src, dst, 1)
                changed = True
    return path


def compose(r1: str, r2: str) -> str | None:
    """Relation of z to x, given r1 = relation of y to x and r2 = z to y."""
    if r1 not in PATHS or r2 not in PATHS:
        return None
    path = reduce_path(PATHS[r1][0] + PATHS[r2][0])
    return BY_PATH.get((path, PATHS[r2][1]))


def fold(relations) -> str | None:
    cur = relations[0]
    for nxt in relations[1:]:
        cur = compose(cur, nxt) if cur else None
        if cur is None:
            return None
    return cur

"""Stage-wise frame validator for RT-NeuS.

Replaces the upstream NeuS-QA ``FrameValidator`` (kept unchanged in
``frame_validator.py`` for the baseline) which mishandles OR and NOT:
pure-OR formulas never pass, an OR clause on one side of ``U`` forces both
sides into the same window, negated propositions are also required by the
AND vote, and only the first ``!`` is honoured.

The formula is split at its top-level ``U`` operators into temporal stages.
A window is valid if it shows evidence of *any* stage, since consecutive
stages of an Until chain happen at different times. Within a stage the
boolean expression is evaluated on the detected propositions:

- ``A | B``  : any child holds
- ``A & B``  : every negated child holds, and a strict majority of the
               positive children hold (the upstream soft-AND vote, tolerant
               to a missed VLM detection when there are 3+ conjuncts)
- ``!A``     : A is not detected
- ``F A``    : A is shown here. Inside ``&``, eventual conjuncts need not
               co-occur: ``F A & F B`` (a sequence question, events at
               different times) accepts a window showing A *or* B, while
               present-tense conjuncts still use the majority vote.
- ``G`` / ``X`` are ignored at frame level, as in upstream.
- an Until nested inside another operator, e.g. ``(A U B) | C``, is read
  like ``|`` at frame level: a window showing either side is evidence.

A formula that still fails to parse falls back to a single stage that any
detected proposition of the formula satisfies, so a malformed PULS output
degrades the filter instead of crashing the run.
"""

import logging
import re

from neus_core.video.video_frame import VideoFrame

_TOKEN_RE = re.compile(r'\s*(?:"([^"]*)"|([()&|!])|([\w\']+))')
_UNARY_TEMPORAL = {"F", "G", "X"}


class _Parser:
    """Recursive-descent parser for one stage's boolean expression.

    Precedence (loosest first): ``|`` / nested ``U``, ``&``, unary ``!``/``F``/``G``.
    Nodes are tuples: ("prop", name), ("not", node), ("and", [nodes]),
    ("or", [nodes]).
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def parse(self):
        if not self.tokens:
            return None
        node = self._or()
        if self.pos != len(self.tokens):
            raise ValueError(f"Unexpected token {self.tokens[self.pos]!r}")
        return node

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _or(self):
        children = [self._and()]
        while self._peek() in (("op", "|"), ("op", "U")):
            self.pos += 1
            children.append(self._and())
        return children[0] if len(children) == 1 else ("or", children)

    def _and(self):
        children = [self._unary()]
        while self._peek() == ("op", "&"):
            self.pos += 1
            children.append(self._unary())
        return children[0] if len(children) == 1 else ("and", children)

    def _unary(self):
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        self.pos += 1
        if tok == ("op", "!"):
            return ("not", self._unary())
        if tok == ("temporal", "F"):
            return ("eventually", self._unary())
        if tok[0] == "temporal":
            return self._unary()
        if tok == ("op", "("):
            node = self._or()
            if self._peek() != ("op", ")"):
                raise ValueError("Unbalanced parentheses")
            self.pos += 1
            return node
        if tok[0] == "prop":
            return ("prop", tok[1])
        raise ValueError(f"Unexpected token {tok!r}")


def _tokenize(expr):
    tokens = []
    pos = 0
    expr = expr.strip()
    while pos < len(expr):
        match = _TOKEN_RE.match(expr, pos)
        if not match or match.end() == pos:
            raise ValueError(f"Cannot tokenize {expr[pos:]!r}")
        quoted, op, word = match.groups()
        if quoted is not None:
            tokens.append(("prop", quoted))
        elif op is not None:
            tokens.append(("op", op))
        elif word == "U":
            tokens.append(("op", "U"))
        elif word in _UNARY_TEMPORAL:
            tokens.append(("temporal", word))
        else:
            tokens.append(("prop", word))
        pos = match.end()
    return tokens


def _strip_outer_parens(expr):
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        for i, ch in enumerate(expr):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and i < len(expr) - 1:
                return expr  # the opening paren closes before the end
        expr = expr[1:-1].strip()
    return expr


def split_top_level_until(expr):
    """Split a formula into temporal stages at top-level `` U `` operators."""
    expr = _strip_outer_parens(expr)
    stages, depth, start, in_quote = [], 0, 0, False
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == '"':
            in_quote = not in_quote
        elif not in_quote:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif depth == 0 and expr[i:i + 3] == " U ":
                stages.append(expr[start:i])
                start = i + 3
                i += 3
                continue
        i += 1
    stages.append(expr[start:])
    if len(stages) == 1:
        return [expr] if expr else []
    # a parenthesised side may hold its own Until chain: a U (b U c)
    return [sub for s in stages if s.strip() for sub in split_top_level_until(s)]


def _collect_props(node, negated=False, positive=None, negative=None):
    positive = set() if positive is None else positive
    negative = set() if negative is None else negative
    kind = node[0]
    if kind == "prop":
        (negative if negated else positive).add(node[1])
    elif kind == "not":
        _collect_props(node[1], not negated, positive, negative)
    elif kind == "eventually":
        _collect_props(node[1], negated, positive, negative)
    else:
        for child in node[1]:
            _collect_props(child, negated, positive, negative)
    return positive, negative


def _evaluate(node, detected):
    kind = node[0]
    if kind == "prop":
        return node[1] in detected
    if kind == "not":
        return not _evaluate(node[1], detected)
    if kind == "eventually":
        return _evaluate(node[1], detected)
    if kind == "or":
        return any(_evaluate(child, detected) for child in node[1])
    # "and": negations are hard constraints, present-tense positives use the
    # majority vote, eventual (F) conjuncts may each be shown at a different time
    hard = [child for child in node[1] if child[0] == "not"]
    eventual = [child for child in node[1] if child[0] == "eventually"]
    soft = [child for child in node[1] if child[0] not in ("not", "eventually")]
    if not all(_evaluate(child, detected) for child in hard):
        return False
    if soft and sum(_evaluate(child, detected) for child in soft) * 2 > len(soft):
        return True
    if any(_evaluate(child, detected) for child in eventual):
        return True
    return not soft and not eventual


class StageFrameValidator:
    def __init__(
        self,
        ltl_formula: str,
        threshold_of_probability: float = 0.5,
    ):
        self.threshold_of_probability = threshold_of_probability

        if "[" in ltl_formula and "]" in ltl_formula:
            ltl_formula = ltl_formula[ltl_formula.find("[") + 1:ltl_formula.rfind("]")]

        try:
            self.stages = [
                _Parser(_tokenize(stage)).parse() for stage in split_top_level_until(ltl_formula)
            ]
            self.stages = [stage for stage in self.stages if stage is not None]
        except ValueError as e:
            props = re.findall(r'"([^"]*)"', ltl_formula)
            logging.warning(f"Could not parse specification {ltl_formula!r} ({e}); "
                            f"falling back to a single any-proposition stage")
            self.stages = [("or", [("prop", p) for p in props])] if props else []
        self.stage_props = [_collect_props(stage) for stage in self.stages]

    def validate_frame(
        self,
        frame: VideoFrame,
    ):
        """Validate frame."""
        thresholded_objects = frame.thresholded_detected_objects(self.threshold_of_probability)
        if len(thresholded_objects) > 0:
            return self.symbolic_verification(frame)
        else:
            return False

    def detected_props(self, frame: VideoFrame):
        detected = set()
        for prop, obj in frame.object_of_interest.items():
            try:
                if obj.get_detected_probability() >= self.threshold_of_probability:
                    detected.add(prop)
            except (AttributeError, TypeError):
                continue
        return detected

    def symbolic_verification(self, frame: VideoFrame):
        """A window is valid if it satisfies at least one temporal stage."""
        detected = self.detected_props(frame)
        return any(_evaluate(stage, detected) for stage in self.stages)

    def stage_of(self, prop):
        """Index of the first stage where ``prop`` appears un-negated, else None."""
        for idx, (positive, _) in enumerate(self.stage_props):
            if prop in positive:
                return idx
        return None

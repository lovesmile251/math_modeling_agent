from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from docx.oxml import OxmlElement
from docx.oxml.ns import qn


_GREEK_AND_SYMBOLS = {
    "alpha": "\u03b1",
    "beta": "\u03b2",
    "gamma": "\u03b3",
    "delta": "\u03b4",
    "epsilon": "\u03b5",
    "varepsilon": "\u03b5",
    "zeta": "\u03b6",
    "eta": "\u03b7",
    "theta": "\u03b8",
    "vartheta": "\u03d1",
    "iota": "\u03b9",
    "kappa": "\u03ba",
    "lambda": "\u03bb",
    "mu": "\u03bc",
    "nu": "\u03bd",
    "xi": "\u03be",
    "pi": "\u03c0",
    "rho": "\u03c1",
    "sigma": "\u03c3",
    "tau": "\u03c4",
    "upsilon": "\u03c5",
    "phi": "\u03c6",
    "varphi": "\u03d5",
    "chi": "\u03c7",
    "psi": "\u03c8",
    "omega": "\u03c9",
    "Gamma": "\u0393",
    "Delta": "\u0394",
    "Theta": "\u0398",
    "Lambda": "\u039b",
    "Xi": "\u039e",
    "Pi": "\u03a0",
    "Sigma": "\u03a3",
    "Phi": "\u03a6",
    "Psi": "\u03a8",
    "Omega": "\u03a9",
    "cdot": "\u00b7",
    "times": "\u00d7",
    "div": "\u00f7",
    "pm": "\u00b1",
    "mp": "\u2213",
    "le": "\u2264",
    "ge": "\u2265",
    "ne": "\u2260",
    "approx": "\u2248",
    "equiv": "\u2261",
    "propto": "\u221d",
    "infty": "\u221e",
    "partial": "\u2202",
    "nabla": "\u2207",
    "in": "\u2208",
    "notin": "\u2209",
    "cap": "\u2229",
    "cup": "\u222a",
    "setminus": "\u2216",
    "sum": "\u2211",
    "prod": "\u220f",
    "int": "\u222b",
    "oint": "\u222e",
    "ldots": "\u2026",
    "cdots": "\u22ef",
    "therefore": "\u2234",
    "because": "\u2235",
}

_NORMAL_OPERATORS = {
    "arg": "arg",
    "max": "max",
    "min": "min",
    "log": "log",
    "ln": "ln",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "exp": "exp",
}

_BLACKBOARD = {
    "E": "\U0001d53c",
    "N": "\u2115",
    "R": "\u211d",
    "Q": "\u211a",
    "Z": "\u2124",
    "C": "\u2102",
}


def add_math_to_paragraph(paragraph, latex: str) -> bool:
    """Append a native Word equation (OMML) to a python-docx paragraph.

    Returns ``False`` only when the input is empty or conversion fails, letting
    callers fall back to plain text. The accepted input is Word-ready LaTeX-like
    equation text without document-level delimiters.
    """

    try:
        omath = latex_to_omml(latex)
    except Exception:
        return False
    paragraph._p.append(omath)
    return True


def latex_to_omml(latex: str):
    expr = _normalise_latex(latex)
    if not expr:
        raise ValueError("empty equation")
    parser = _LatexParser(expr)
    omath = _m("oMath")
    _extend(omath, parser.parse())
    return omath


def _normalise_latex(latex: str) -> str:
    expr = latex.strip()
    for left, right in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if expr.startswith(left) and expr.endswith(right):
            expr = expr[len(left) : -len(right)].strip()
            break
    expr = expr.replace("路", r"\cdot ").replace("·", r"\cdot ")
    expr = expr.replace(r"\left", "").replace(r"\right", "")
    expr = expr.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "")
    expr = expr.replace(r"\quad", " ").replace(r"\qquad", " ")
    return expr.strip()


def _m(tag: str):
    return OxmlElement(f"m:{tag}")


def _extend(parent, children: Iterable) -> None:
    for child in children:
        parent.append(child)


def _run(text: str, *, normal: bool | None = None):
    r = _m("r")
    if normal is True or (normal is None and (len(text) > 1 or not text.isalpha())):
        rpr = _m("rPr")
        rpr.append(_m("nor"))
        r.append(rpr)
    t = _m("t")
    t.text = text
    r.append(t)
    return r


def _fraction(num_children: list, den_children: list):
    f = _m("f")
    num = _m("num")
    den = _m("den")
    _extend(num, num_children)
    _extend(den, den_children)
    f.append(num)
    f.append(den)
    return f


def _script(base_children: list, sub_children: list | None, sup_children: list | None):
    if sub_children is not None and sup_children is not None:
        node = _m("sSubSup")
        e = _m("e")
        sub = _m("sub")
        sup = _m("sup")
        _extend(e, base_children)
        _extend(sub, sub_children)
        _extend(sup, sup_children)
        node.append(e)
        node.append(sub)
        node.append(sup)
        return node
    if sub_children is not None:
        node = _m("sSub")
        e = _m("e")
        sub = _m("sub")
        _extend(e, base_children)
        _extend(sub, sub_children)
        node.append(e)
        node.append(sub)
        return node
    node = _m("sSup")
    e = _m("e")
    sup = _m("sup")
    _extend(e, base_children)
    _extend(sup, sup_children or [])
    node.append(e)
    node.append(sup)
    return node


def _radical(value_children: list, degree_children: list | None = None):
    rad = _m("rad")
    if degree_children:
        deg = _m("deg")
        _extend(deg, degree_children)
        rad.append(deg)
    else:
        rad_pr = _m("radPr")
        deg_hide = _m("degHide")
        deg_hide.set(qn("m:val"), "1")
        rad_pr.append(deg_hide)
        rad.append(rad_pr)
    e = _m("e")
    _extend(e, value_children)
    rad.append(e)
    return rad


@dataclass
class _LatexParser:
    text: str
    pos: int = 0

    def parse(self, stop: str | None = None) -> list:
        out: list = []
        while self.pos < len(self.text):
            if stop is not None and self.text.startswith(stop, self.pos):
                self.pos += len(stop)
                break
            ch = self.text[self.pos]
            if stop is None and ch == "}":
                self.pos += 1
                break
            if ch.isspace():
                self.pos += 1
                continue
            atom = self._atom()
            if not atom:
                continue
            out.extend(self._with_scripts(atom))
        return out

    def _with_scripts(self, base: list) -> list:
        sub = None
        sup = None
        while self.pos < len(self.text) and self.text[self.pos] in "_^":
            kind = self.text[self.pos]
            self.pos += 1
            script = self._script_arg()
            if kind == "_":
                sub = script
            else:
                sup = script
        if sub is None and sup is None:
            return base
        return [_script(base, sub, sup)]

    def _script_arg(self) -> list:
        if self.pos < len(self.text) and self.text[self.pos] == "{":
            return self._required_group()
        return self._atom()

    def _atom(self) -> list:
        if self.pos >= len(self.text):
            return []
        ch = self.text[self.pos]
        if ch == "{":
            return self._required_group()
        if ch == "\\":
            return self._command()
        if ch in "([{":
            return self._delimited(ch)
        if ch in ")]}":
            self.pos += 1
            return [_run(ch)]
        self.pos += 1
        return [_run(ch)]

    def _delimited(self, opener: str) -> list:
        closer = {"(": ")", "[": "]", "{": "}"}[opener]
        self.pos += 1
        children = self.parse(closer)
        return [_run(opener), *children, _run(closer)]

    def _command(self) -> list:
        self.pos += 1
        if self.pos >= len(self.text):
            return [_run("\\")]
        if not self.text[self.pos].isalpha():
            ch = self.text[self.pos]
            self.pos += 1
            if ch in "{}":
                return [_run(ch)]
            return [_run(ch)]

        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isalpha():
            self.pos += 1
        name = self.text[start : self.pos]

        if name == "frac":
            return [_fraction(self._required_group(), self._required_group())]
        if name == "sqrt":
            degree = self._optional_bracket_group()
            return [_radical(self._required_group(), degree)]
        if name in {"mathrm", "operatorname", "text"}:
            return [_run(self._read_group_text(), normal=True)]
        if name == "mathbb":
            literal = self._read_group_text()
            return [_run("".join(_BLACKBOARD.get(ch, ch) for ch in literal), normal=True)]
        if name in {"mathbf", "boldsymbol", "vec", "hat", "bar", "dot", "ddot"}:
            return self._required_group()
        if name in _GREEK_AND_SYMBOLS:
            return [_run(_GREEK_AND_SYMBOLS[name])]
        if name in _NORMAL_OPERATORS:
            return [_run(_NORMAL_OPERATORS[name], normal=True)]
        return [_run(name, normal=True)]

    def _required_group(self) -> list:
        if self.pos >= len(self.text) or self.text[self.pos] != "{":
            return self._atom()
        self.pos += 1
        return self.parse("}")

    def _optional_bracket_group(self) -> list | None:
        if self.pos >= len(self.text) or self.text[self.pos] != "[":
            return None
        self.pos += 1
        return self.parse("]")

    def _read_group_text(self) -> str:
        if self.pos >= len(self.text) or self.text[self.pos] != "{":
            return ""
        self.pos += 1
        depth = 1
        start = self.pos
        while self.pos < len(self.text) and depth:
            ch = self.text[self.pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    raw = self.text[start : self.pos]
                    self.pos += 1
                    return _clean_literal(raw)
            self.pos += 1
        return _clean_literal(self.text[start : self.pos])


def _clean_literal(raw: str) -> str:
    raw = raw.replace(r"\ ", " ")
    raw = re.sub(r"\\([{}])", r"\1", raw)
    return raw

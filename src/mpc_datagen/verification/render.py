import math
import re

import numpy as np
from rich.console import Console
from rich.table import Table

from .reports import StabilityReport


def pretty_num(
    value: float,
    *,
    sci_small: float = 1e-3,
    sci_large: float = 1e4,
    max_decimals: int = 6,
    max_sci_decimals: int = 6,
) -> str:
    """Format numbers for human-readable report display.

    Examples
    --------
    - 1.000       -> "1"
    - 1.000e-04   -> "1e-4"
    - 9.1771      -> "9.1771"
    """
    x = float(value)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"

    # Near-integer -> integer
    if np.isclose(x, round(x), rtol=0.0, atol=1e-12 * max(1.0, abs(x))):
        return str(int(round(x)))

    ax = abs(x)
    use_sci = (ax != 0.0) and (ax < sci_small or ax >= sci_large)
    if use_sci:
        s = f"{x:.{max_sci_decimals}e}"
        mantissa, exp = s.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp_int = int(exp)
        return f"{mantissa}e{exp_int}"

    s = f"{x:.{max_decimals}f}"
    s = s.rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


_NUM_RE = re.compile(
    r"(?<![\w])"  # don't start in the middle of an identifier
    r"[+-]?"  # sign
    r"(?:"  # number
    r"(?:\d+\.\d+|\d+\.|\.\d+|\d+)"  # int/float
    r"(?:[eE][+-]?\d+)?"  # optional exponent
    r")"
    r"(?![\w])"  # don't end in the middle of an identifier
)


def prettify_text(text: str) -> str:
    """Replace numeric substrings in text with pretty_num formatting."""

    def _repl(match: re.Match) -> str:
        token = match.group(0)
        # Skip plain integers
        if re.fullmatch(r"[+-]?\d+", token):
            return token
        try:
            return pretty_num(float(token))
        except ValueError:
            return token

    return _NUM_RE.sub(_repl, text)


class VerificationRender(Table):
    def __init__(self, report: StabilityReport):
        super().__init__(title=report.method + " Report")
        self.add_column("Check")
        self.add_column("Result")
        self.add_column("Details")
        self.add_row("Overall Stability", str(report.is_stable), prettify_text(report.message))
        
        for rep in report.details.values():
            if isinstance(rep, StabilityReport):
                self.add_row(rep.method, str(rep.is_stable), prettify_text(rep.message))

    def render(self):
        console = Console()
        console.print(self)
    
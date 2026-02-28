"""ui/colors.py — colour helper functions wrapping ANSI codes."""

from config import (RESET, BOLD, DIM, ITALIC, UNDER, FG, BG,
                    C_HEADER, C_FORMULA, C_SUCCESS, C_ERROR,
                    C_WARN, C_BLOCK, C_BODY, C_HINT, C_CODE,
                    C_VALUE, C_SECTION, C_EMPH)


def _c(code: str, text: str) -> str:
    return f"{code}{text}{RESET}"


def bold(t):       return _c(BOLD, t)
def dim(t):        return _c(DIM, t)
def italic(t):     return _c(ITALIC, t)
def underline(t):  return _c(UNDER, t)

def cyan(t):       return _c(FG["cyan"], t)
def yellow(t):     return _c(FG["yellow"], t)
def green(t):      return _c(FG["green"], t)
def red(t):        return _c(FG["red"], t)
def blue(t):       return _c(FG["blue"], t)
def magenta(t):    return _c(FG["magenta"], t)
def white(t):      return _c(FG["white"], t)
def grey(t):       return _c(FG["grey"], t)

def bcyan(t):      return _c(FG["bcyan"], t)
def byellow(t):    return _c(FG["byellow"], t)
def bgreen(t):     return _c(FG["bgreen"], t)
def bred(t):       return _c(FG["bred"], t)
def bblue(t):      return _c(FG["bblue"], t)
def bmagenta(t):   return _c(FG["bmagenta"], t)
def bwhite(t):     return _c(FG["bwhite"], t)

def header(t):     return _c(C_HEADER, t)
def formula(t):    return _c(C_FORMULA, t)
def success(t):    return _c(C_SUCCESS, t)
def error(t):      return _c(C_ERROR, t)
def warn(t):       return _c(C_WARN, t)
def block_title(t):return _c(C_BLOCK, t)
def body(t):       return _c(C_BODY, t)
def hint(t):       return _c(C_HINT, t)
def code_color(t): return _c(C_CODE, t)
def value(t):      return _c(C_VALUE, t)
def section(t):    return _c(C_SECTION, t)
def emph(t):       return _c(C_EMPH, t)

def bold_cyan(t):     return _c(FG["cyan"] + BOLD, t)
def bold_yellow(t):   return _c(FG["yellow"] + BOLD, t)
def bold_green(t):    return _c(FG["green"] + BOLD, t)
def bold_magenta(t):  return _c(FG["magenta"] + BOLD, t)
def bold_red(t):      return _c(FG["red"] + BOLD, t)

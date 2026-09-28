"""Text normalisation for business names and addresses.

Everything here is rule-based and country-agnostic: transliterate to ASCII
(handles accents and Devanagari), lowercase, strip punctuation, expand common
abbreviations (EN/IN/FR), and separate legal-form tokens from the name core.
"""
import re
from functools import lru_cache

from unidecode import unidecode

# abbreviation -> canonical token (applied per token after punctuation removal)
ABBR = {
    # legal forms
    "pvt": "private", "prvt": "private", "praivet": "private", "pravet": "private", "limitted": "limited", "pte": "private", "ltd": "limited", "ltd.": "limited",
    "corp": "corporation", "co": "company", "cos": "company", "inc": "incorporated",
    "incorp": "incorporated", "assn": "association", "assoc": "association",
    "intl": "international", "mfg": "manufacturing", "svc": "service", "svcs": "services",
    "bros": "brothers", "cie": "compagnie", "ste": "societe", "sa": "sa",
    # street types (EN)
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "pkwy": "parkway", "hwy": "highway", "tpke": "turnpike", "tcke": "turnpike",
    "cir": "circle", "trl": "trail", "sq": "square", "ter": "terrace", "ste.": "suite",
    "apt": "apartment", "fl": "floor", "bldg": "building", "mt": "mount", "ft": "fort",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    # street types (FR)
    "r": "rue", "imp": "impasse", "chem": "chemin", "rte": "route",
    "fbg": "faubourg", "pte": "porte", "qu": "quai",
    # India
    "opp": "opposite", "nr": "near", "mkt": "market", "ngr": "nagar", "clny": "colony",
    "dist": "district", "vill": "village", "po": "post", "hno": "house",
}

# tokens carrying no identity information
STOP = {"the", "and", "of", "na", "null", "none", "nan", "no", "number", "m", "ms", "dba",
        "doing", "business", "as", "aka", "de", "du", "des", "la", "le", "les", "et", "d", "l"}

# legal form / generic business-type words, removed from the name core
LEGAL = {
    "private", "limited", "llc", "llp", "lp", "pllc", "pc", "corporation", "company",
    "incorporated", "plc", "gmbh", "sarl", "sas", "sasu", "eurl", "sci", "snc", "sa",
    "compagnie", "societe", "group", "groupe", "holdings", "enterprises", "partners",
    "center", "centre", "acquisition", "services", "solutions", "fils", "freres",
}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "newhampshire",
    "nj": "newjersey", "nm": "newmexico", "ny": "newyork", "nc": "northcarolina",
    "nd": "northdakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhodeisland", "sc": "southcarolina", "sd": "southdakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "westvirginia", "wi": "wisconsin", "wy": "wyoming", "dc": "districtofcolumbia",
}
IN_STATES = {
    "ap": "andhrapradesh", "ar": "arunachalpradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachalpradesh", "jh": "jharkhand", "ka": "karnataka", "kl": "kerala",
    "mp": "madhyapradesh", "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya",
    "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "odisha", "orissa": "odisha",
    "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamilnadu", "ts": "telangana",
    "tg": "telangana", "tr": "tripura", "up": "uttarpradesh", "uk": "uttarakhand",
    "ua": "uttarakhand", "wb": "westbengal", "dl": "delhi", "jk": "jammukashmir",
    "ch": "chandigarh", "py": "puducherry", "an": "andamannicobar", "ld": "lakshadweep",
    "la": "ladakh", "dn": "dadranagarhaveli", "dd": "damandiu",
}
# multi-word state names collapsed to one token so they match the abbreviations above
MULTI = [(re.compile(r"\b" + " ".join(v_parts) + r"\b"), "".join(v_parts)) for v_parts in (
    ("new", "york"), ("new", "jersey"), ("new", "mexico"), ("new", "hampshire"),
    ("north", "carolina"), ("south", "carolina"), ("north", "dakota"), ("south", "dakota"),
    ("west", "virginia"), ("rhode", "island"), ("district", "of", "columbia"),
    ("andhra", "pradesh"), ("arunachal", "pradesh"), ("himachal", "pradesh"),
    ("madhya", "pradesh"), ("uttar", "pradesh"), ("tamil", "nadu"), ("west", "bengal"),
    ("jammu", "and", "kashmir"), ("jammu", "kashmir"),
)]

_nonalnum = re.compile(r"[^a-z0-9 ]+")
_dup = re.compile(r"([a-z])\1+")
_ws = re.compile(r"\s+")
_url = re.compile(r"(?:www\.)?([a-z0-9]+)\.(?:com|in|net|org|co|fr|biz|info)\b")


@lru_cache(maxsize=200_000)
def ascii_lower(s: str) -> str:
    """Transliterate to ASCII, lowercase, keep alnum + spaces, squash doubled letters.

    Doubled-letter squashing makes Devanagari transliterations ('raam') and common
    typos ('Houstoon') line up with their plain English forms."""
    s = unidecode(s).lower().replace("&", " and ").replace("@", " at ")
    s = _url.sub(r" \1 ", s)
    s = _nonalnum.sub(" ", s)
    for pat, rep in MULTI:
        s = pat.sub(rep, s)
    s = _dup.sub(r"\1", s)
    return _ws.sub(" ", s).strip()


def tokens(s: str, country: str = "", address: bool = False) -> list:
    """Normalised token list with abbreviations expanded and stop tokens removed.

    For addresses, a comma-separated segment that is exactly one state code is
    mapped to the full state name (so 'CT' is Connecticut there, 'Ct' Court elsewhere)."""
    states = US_STATES if country == "US" else IN_STATES if country == "India" else {}
    out = []
    for seg in (s.split(",") if address else [s]):
        toks = ascii_lower(seg).split()
        if address and len(toks) == 1 and toks[0] in states:
            out.append(states[toks[0]])
            continue
        for t in toks:
            t = ABBR.get(t, t)
            if t not in STOP:
                out.append(t)
    return out


def name_parts(name: str, country: str = ""):
    """Return (core, legal) token lists for a business name."""
    toks = tokens(name, country)
    core = [t for t in toks if t not in LEGAL]
    legal = [t for t in toks if t in LEGAL]
    return core, legal


_num = re.compile(r"\d+")


def addr_numbers(addr: str) -> list:
    """All digit runs in the address (house numbers, PIN/ZIP codes)."""
    return _num.findall(unidecode(addr))


if __name__ == "__main__":
    assert ascii_lower("Houstoon") == "houston"
    assert name_parts("OPX India Private Ltd.", "India") == (["opx", "india"], ["private", "limited"])
    assert tokens("511 LORINO ST, HOUSTOON, TX", "US", True) == ["511", "lorino", "street", "houston", "texas"]
    assert tokens("63 R. DE DIEPPE, LILLE", "France", True) == ["63", "rue", "diepe", "lile"]
    assert tokens("12 Main Ct, Hartford, CT", "US", True)[-1] == "connecticut"
    print(name_parts("राम मार्केटिंग प्राइवेट लिमिटेड"))
    print("ok")

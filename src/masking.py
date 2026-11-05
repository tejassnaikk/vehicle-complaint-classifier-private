"""Mask contact details that appear inside the complaint text (rule 14 of 03_clean_and_label.py).

Replaced with placeholders: URLs -> [URL], e-mail addresses -> [EMAIL], full street addresses and "city, ST ZIP" endings -> [ADDRESS],
US phone numbers -> [PHONE], VIN-like 17-character strings -> [VIN]. Applied to lowercased text, in this order (URLs first, because a URL
can contain a VIN or an e-mail-like part).

Precision notes (the text is lowercase, so capital letters cannot help):
- A street address needs a house number, one to three alphabetic name words that are not common non-street words ("70 mph on the road"
  is not an address), and a street suffix (st, ave, rd, blvd, ...). City, state and ZIP are masked with it when they follow.
- A bare "city, ST ZIP" ending needs a comma before the state, and the state abbreviations that are also English words
  (in, or, me, id, hi, oh, ok) are only accepted inside a full street address, so "in 20000 miles" is never touched.
- A phone number needs separators (spaces, dots or dashes) between the digit groups; a bare 10-digit run is left alone.
"""
import re

STATES = ("al ak az ar ca co ct de dc fl ga ia il ks ky la ma md mi mn ms mo mt ne nv nh nj nm ny nc nd pa ri sc sd tn tx ut vt va wa wv wi wy pr").split()
STATES_AMBIGUOUS = ["in", "or", "me", "id", "hi", "oh", "ok"]  # also English words: only used after a street address
ST = r"(?:" + "|".join(STATES) + r")"
ST_ANY = r"(?:" + "|".join(STATES + STATES_AMBIGUOUS) + r")"
SUFFIX = (r"(?:st|street|ave|avenue|av|rd|road|blvd|boulevard|dr|drive|ln|lane|hwy|highway|pkwy|parkway|way|ct|court|pl|place|cir|circle|"
          r"trl|trail|ter|terrace|sq|square|pike|turnpike|tpke|expy|expressway|fwy|freeway|loop|plaza|plz|crossing|xing|bypass)")
NOT_NAME = (r"(?:mph|miles?|feet|foot|ft|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|yrs?|times?|km|kph|lbs?|pounds?|percent|degrees?|"
            r"the|a|an|in|on|at|to|of|and|or|my|our|for|per|from|with|was|is|it|i|that|this|but|so|as|by|when|while|then|up|down|off|out|"
            r"left|right|interstate|southbound|northbound|eastbound|westbound|toll|during|normal|mostly|still|high|low|regular|single|double|two|three|"
            r"multi|rough|bumpy|dirt|gravel|wet|dry|icy|steep|winding|open|slow|fast|long|short|back|front|rear|side|same|other|those|these|any|one|every|each|all|some|no)")
NAME_WORD = r"(?:\d+(?:st|nd|rd|th)|(?!" + NOT_NAME + r"\b)[a-z][a-z'.-]{2,})"
DIRECTION = r"(?:(?:n|s|e|w|ne|nw|se|sw|north|south|east|west)\.?\s+)?"
UNIT = r"(?:,?\s*(?:apt|suite|ste|unit|bldg|building)\.?\s*#?[a-z0-9-]+)?"
CITY = r"(?:[a-z][a-z.'-]*\s+){0,2}[a-z][a-z.'-]*"
ZIP = r"\d{5}(?:-\d{4})?"

URL_RE = re.compile(r"(?:https?://|www\.)[^\s)\]>\"']+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]*)+")
TAIL = rf",?\s*{CITY},?\s+{ST_ANY}\s+{ZIP}"
STREET_NAMED = rf"{DIRECTION}(?:{NAME_WORD}\s+){{1,3}}?{SUFFIX}\b\.?{UNIT}"
ROUTE = rf"{DIRECTION}(?:rte|route|hwy|highway|sr|us|cr|fm|county road|state route)\.?\s*-?\d{{1,4}}[a-z]?\b{UNIT}"  # "1118 w rte 66", "3061 us-42"
STREET_CORE = rf"(?:{STREET_NAMED}|{ROUTE})"
# Without a city, state and ZIP after it, the house number must be 3 to 6 digits and must not look like a model year (1980 to 2039): "2020 ford
# explorer st" is a trim level, not a street. With a full city, state and ZIP after it, any number is accepted ("2 main st, springfield, il 62704").
YEAR = r"(?:19[89]\d|20[0-3]\d)"
STREET_RE = re.compile(rf"(?<![\d,$.])\b(?!{YEAR}\b)\d{{3,6}}\s+{STREET_CORE}(?:{TAIL})?|\b\d{{1,6}}\s+{STREET_CORE}{TAIL}")
CITY_ST_ZIP_RE = re.compile(rf"\b{CITY},?\s+{ST}\s+{ZIP}\b")
PHONE_RE = re.compile(r"(?<![\w-])(?:\+?1[ .-]?)?(?:\(\d{3}\)[ .-]*|\d{3}[ .-])\d{3}[ .-]+\d{4}(?![\w-])")
VIN_RE = re.compile(r"\b(?=[a-hj-npr-z0-9]*\d)(?=[a-hj-npr-z0-9]*[a-z])[a-hj-npr-z0-9]{17}\b")

RULES = [("[URL]", "url", URL_RE), ("[EMAIL]", "email", EMAIL_RE), ("[ADDRESS]", "address", STREET_RE),
         ("[ADDRESS]", "address", CITY_ST_ZIP_RE), ("[PHONE]", "phone", PHONE_RE), ("[VIN]", "vin", VIN_RE)]


def mask_contact_details(text):
    for placeholder, _, rx in RULES:
        text = rx.sub(placeholder, text)
    return text


def tag_counts(series):
    """Number of texts containing each kind of contact detail (before masking); one pass per kind."""
    out = {}
    for _, kind, rx in RULES:
        out[kind] = out.get(kind, 0) + int(series.str.contains(rx, regex=True).sum())
    # address has two patterns: count texts matching either
    out["address"] = int((series.str.contains(STREET_RE, regex=True) | series.str.contains(CITY_ST_ZIP_RE, regex=True)).sum())
    return out

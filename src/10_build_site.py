"""Phase 8: build the static website in docs/ from files in results/ and docs/img/.

All page text is written here; every number is read from a file in results/ (or computed from those files),
so the site cannot drift from the results. Run: python src/10_build_site.py
Also writes two small analysis files: results/eval/error_overlap.json and results/eval/stub_analysis.json.
"""
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, PLAIN_NAMES, RESULTS_DIR as R, ROOT, CLEAN_DIR, VAL_START, TEST_START, TEST_END

DOCS = ROOT / "docs"
IMG = DOCS / "img"
REPO = "https://github.com/tejassnaikk/vehicle-complaint-classifier"
NHTSA_DATA = "https://www.nhtsa.gov/nhtsa-datasets-and-apis"
NHTSA_DICT = "https://static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt"
NHTSA_FILES = ["https://static.nhtsa.gov/odi/ffdd/cmpl/COMPLAINTS_RECEIVED_2020-2024.zip",
               "https://static.nhtsa.gov/odi/ffdd/cmpl/COMPLAINTS_RECEIVED_2025-2026.zip"]
NHTSA_REPORT = "https://www.nhtsa.gov/report-a-safety-problem"
MODELS = ["mlp", "bilstm", "distilbert"]
NICE = {"mlp": "MLP (TF-IDF)", "bilstm": "BiLSTM", "distilbert": "DistilBERT"}
PAGES = [("introduction.html", "Introduction"), ("data.html", "Data & Prep"), ("models.html", "Models"),
         ("results.html", "Results"), ("conclusion.html", "Conclusion"), ("about.html", "About"), ("try.html", "Explore")]
# Interactive pages, linked from the Explore group in the nav and from the home page. Add a page here when its builder exists.
EXPLORE = [("try.html", "Try it", "Type or load a complaint and watch our simplest model sort it, running in your browser."),
           ]
EXPLORE_FILES = [f for f, _, _ in EXPLORE]
SPLITS = ["temporal", "random"]


# ============================================================================ helpers
def jl(p):
    return json.loads(Path(p).read_text())


def n(x):
    return f"{int(round(x)):,}"


def p1(x):
    return f"{x:.1f}"


def f4(x):
    return f"{x:.4f}"


def f3(x):
    return f"{x:.3f}"


def esc(s):
    return html.escape(str(s), quote=True)


def para(sentences, strict=False):
    """One paragraph from a list of sentences. strict=True enforces the 7 to 10 sentence rule."""
    if strict:
        assert 7 <= len(sentences) <= 10, f"paragraph has {len(sentences)} sentences: {sentences[0][:50]}"
    return "<p>" + " ".join(sentences) + "</p>\n"


def zoomimg(src, alt):
    with Image.open(IMG / src) as im:
        w, h = im.size
    return f'<a class="zoom" href="img/{src}" title="Open the full-size image"><img src="img/{src}" alt="{esc(alt)}" width="{w}" height="{h}" loading="lazy"></a>'


def fig(src, alt, caption, extra=""):
    return f'<figure class="figcard"{extra}>{zoomimg(src, alt)}<figcaption>{caption}</figcaption></figure>\n'


def table(headers, rows, caption=None, num_cols=(), best_rows=(), text_cols=()):
    t = '<div class="table-wrap"><table>'
    if caption:
        t += f"<caption>{caption}</caption>"
    t += "<thead><tr>" + "".join(f'<th scope="col"{" class=num" if i in num_cols else ""}>{h}</th>' for i, h in enumerate(headers)) + "</tr></thead><tbody>"
    for ri, row in enumerate(rows):
        t += "<tr" + (' class="best"' if ri in best_rows else "") + ">"
        for i, c in enumerate(row):
            cls = "num" if i in num_cols else ("text" if i in text_cols else "")
            t += f'<td{" class=" + cls if cls else ""}>{c}</td>'
        t += "</tr>"
    return t + "</tbody></table></div>\n"


def details(summary, body, cls=""):
    return f'<details{" class=" + cls if cls else ""}><summary>{summary}</summary>\n{body}</details>\n'


def tech(summary, items):
    """Collapsible technical details (bullets only) so the main text stays non-technical."""
    return details(summary, "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>", "tech")


def gh(path):
    return f"{REPO}/blob/main/{path}"


def callout(title, body_html, kind=""):
    return f'<aside class="callout {kind}"><div class="callout-title">{esc(title)}</div><div class="body">{body_html}</div></aside>\n'


def pull(text):
    return f'<div class="pull">{text}</div>\n'


def pill(text, kind=""):
    return f'<span class="pill {kind}">{esc(text)}</span>'


def stat(num, label, error=False):
    return f'<div class="stat{" is-error" if error else ""}"><div class="num">{num}</div><div class="label">{label}</div></div>'


def ticket(q):
    return (f'<article class="ticket"><div class="ticket-head"><span class="tag">{esc(q["plain_label"])}</span>'
            f'<span class="veh">{esc(q["year"])} {esc(q["make"])}</span></div>'
            f'<div class="ticket-body"><blockquote>&ldquo;{esc(q["excerpt"])}&rdquo;</blockquote></div>'
            f'<div class="ticket-foot"><span>REPORT #{esc(q["odino"])}</span><span>Received {esc(q["received"])}</span></div></article>\n')


ORDER = [("index.html", "Home"), ("introduction.html", "Introduction"), ("data.html", "Data & Prep"), ("models.html", "Models"),
         ("results.html", "Results"), ("conclusion.html", "Conclusion"), ("about.html", "About")]
STEPS = [("introduction.html", "Introduction"), ("data.html", "Data"), ("models.html", "Models"), ("results.html", "Results"), ("conclusion.html", "Conclusion")]


def phead(kicker, title, subtitle=None):
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    return f'<div class="page-head"><div class="inner"><div class="kicker">{esc(kicker)}</div><h1>{esc(title)}</h1>{sub}</div></div>\n'


def pager(fname):
    if fname in EXPLORE_FILES:
        i = EXPLORE_FILES.index(fname)
        pv = EXPLORE[i - 1][:2] if i > 0 else ("index.html", "Home")
        prev = f'<a class="prev" href="{pv[0]}"><span class="dir">&larr; Previous</span><span class="ttl">{esc(pv[1])}</span></a>'
        nxt = (f'<a class="next" href="{EXPLORE[i + 1][0]}"><span class="dir">Next &rarr;</span><span class="ttl">{esc(EXPLORE[i + 1][1])}</span></a>'
               if i < len(EXPLORE) - 1 else '<a class="next" href="results.html"><span class="dir">Back to the story</span><span class="ttl">Results</span></a>')
        return f'<nav class="pager" aria-label="Previous and next page">{prev}{nxt}</nav>\n'
    names = [f for f, _ in ORDER]
    i = names.index(fname)
    prev = f'<a class="prev" href="{ORDER[i - 1][0]}"><span class="dir">&larr; Previous</span><span class="ttl">{esc(ORDER[i - 1][1])}</span></a>' if i > 0 else ""
    nxt = f'<a class="next" href="{ORDER[i + 1][0]}"><span class="dir">Next &rarr;</span><span class="ttl">{esc(ORDER[i + 1][1])}</span></a>' if i < len(ORDER) - 1 else ""
    return f'<nav class="pager" aria-label="Previous and next page">{prev}{nxt}</nav>\n'


def page(fname, title, body, desc, head="", scripts=()):
    on_explore = fname in EXPLORE_FILES
    nav = "".join(f'<li><a href="{f}"{" aria-current=\"page\"" if (f == fname or (t == "Explore" and on_explore)) else ""}{" class=explore-link" if t == "Explore" else ""}>{esc(t)}</a></li>'
                  for f, t in PAGES if t != "Explore" or EXPLORE)
    if on_explore:
        strip = ('<nav class="explore-bar" aria-label="Explore"><div class="inner"><span class="lab">Explore</span><ul>'
                 + "".join(f'<li><a href="{f}"{" aria-current=\"page\"" if f == fname else ""}>{esc(t)}</a></li>' for f, t, _ in EXPLORE) + '</ul></div></nav>')
    else:
        strip = None
    extra = "".join(f'<script src="js/{x}" defer></script>\n' for x in scripts)
    prog = "".join(f'<li><a href="{f}"{" aria-current=\"step\"" if f == fname else ""}><span class="n">{i}</span><span class="t">{esc(t)}</span></a></li>'
                   for i, (f, t) in enumerate(STEPS, 1))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#1c1f24">
<title>{esc(title)} | What Failed on This Car?</title>
<meta name="description" content="{esc(desc)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&amp;family=Inter:wght@400;500;600;700&amp;family=JetBrains+Mono:wght@400;500&amp;display=swap">
<link rel="stylesheet" href="css/style.css">
<script src="js/site.js" defer></script>
{extra}</head>
<body>
<a class="skip" href="#content">Skip to the content</a>
<header class="site-header"><div class="bar">
<a class="brand" href="index.html"><span class="brand-mark" aria-hidden="true"></span>What Failed on This Car?</a>
<nav class="main" aria-label="Main"><ul>{nav}</ul></nav>
</div></header>
{strip if strip else '<nav class="progress" aria-label="Progress through the site"><ol>' + prog + '</ol></nav>'}
{head}<main id="content">
{body}
{pager(fname)}</main>
<footer><div class="inner">
<div>Complaint data: National Highway Traffic Safety Administration (NHTSA), Office of Defects Investigation, vehicle owner complaints. US government data, public domain.</div>
<div>A class project. Code, cleaned data and results: <a href="{REPO}">{REPO.replace("https://", "")}</a>.</div>
</div></footer>
</body>
</html>
"""


# ============================================================================ load results
funnel = jl(R / "data_funnel.json")
facts = jl(R / "eda" / "dataset_facts.json")
per_year = pd.read_csv(R / "eda" / "complaints_per_year.csv")
PY = {int(r.year): int(r.complaints) for r in per_year.itertuples()}
summ = pd.read_csv(R / "summary_table.csv")
sig = jl(R / "significance.json")
trans = jl(R / "transcript_vs_owner.json")
cb = pd.read_csv(R / "class_balance.csv").set_index("class")
pcf1 = pd.read_csv(R / "eval" / "per_class_f1_all.csv", index_col=0)
ident = pd.read_csv(R / "figures" / "data" / "identified_correctly.csv").set_index("class").percent_sorted_correctly
mix = pd.read_csv(R / "figures" / "data" / "mixups.csv")
quotes = jl(R / "figures" / "data" / "quote_cards.json")
cfg = {(m, s): jl(R / m / s / "config.json") for m in MODELS for s in SPLITS}
tm = {(m, s): jl(R / m / s / "metrics.json") for m in MODELS for s in SPLITS}
em = {(m, s): jl(R / "eval" / m / s / "metrics.json") for m in MODELS for s in SPLITS}
arch = jl(R / "distilbert" / "architecture.json")
wi = jl(R / "eval" / "word_importance" / "word_importance.json")
audit_after = (R / "leakage_audit.txt").read_text()
audit_before = (R / "leakage_audit_before_stripping.txt").read_text()
rules_log = (R / "cleaning_rules_log.txt").read_text()


def S(model, split):
    return summ[(summ.model == NICE[model]) & (summ.split == split)].iloc[0]


# ---- software versions (pinned in requirements.txt), saved to results/environment.json
env = dict(line.strip().split("==") for line in (ROOT / "requirements.txt").read_text().split() if "==" in line)
env["python"] = f"{sys.version_info.major}.{sys.version_info.minor}"
(R / "environment.json").write_text(json.dumps(env, indent=2))

# ---- extra analyses saved to results/eval
def load_preds(m, s):
    return pd.read_csv(R / m / s / "test_predictions.csv", dtype={"odino": str})


test_t = pd.read_csv(CLEAN_DIR / "complaints_temporal.csv", dtype=str, keep_default_na=False)
test_t = test_t[test_t.split == "test"].reset_index(drop=True)
P = {m: load_preds(m, "temporal").merge(test_t[["odino", "text"]], on="odino") for m in MODELS}
wrong = {m: (P[m].y_true != P[m].y_pred) for m in MODELS}
overlap = {"n_test": int(len(test_t)), "wrong_by_model": {m: int(wrong[m].sum()) for m in MODELS},
           "wrong_for_all_three": int((wrong["mlp"] & wrong["bilstm"] & wrong["distilbert"]).sum()),
           "wrong_for_at_least_one": int((wrong["mlp"] | wrong["bilstm"] | wrong["distilbert"]).sum())}
overlap["pct_wrong_for_all_three"] = overlap["wrong_for_all_three"] / overlap["n_test"] * 100
(R / "eval" / "error_overlap.json").write_text(json.dumps(overlap, indent=2))
STUB = re.compile(r"notification of\s*[;.,]")
has_stub = P["mlp"].text.str.contains(STUB)
stub = {"description": "temporal test texts that contain the stub 'notification of;' left after the NHTSA campaign number and component name were removed",
        "n_test": int(len(has_stub)), "n_with_stub": int(has_stub.sum()),
        "accuracy": {m: {"with_stub": float((~wrong[m])[has_stub].mean()), "without_stub": float((~wrong[m])[~has_stub].mean())} for m in MODELS}}
(R / "eval" / "stub_analysis.json").write_text(json.dumps(stub, indent=2))

# ---- parse the cleaning rule log: rule number -> name, count, examples
rules = {}
for block in re.split(r"(?m)^### ", rules_log)[1:]:
    head, *lines = block.strip().splitlines()
    m_ = re.match(r"(\d+) (.+?)\s+\(([\d,]+) texts changed\)", head)
    ex, before = [], None
    for ln in lines:
        if ln.strip().startswith("BEFORE:"):
            before = ln.split("BEFORE:", 1)[1].strip()
        elif ln.strip().startswith("AFTER"):
            ex.append((before, ln.split(":", 1)[1].strip()))
    rules[int(m_.group(1))] = {"name": m_.group(2), "count": int(m_.group(3).replace(",", "")), "examples": ex}

# ---- parse the "before stripping" audit: the terms that gave the leak away
leak_terms = []
for term in ["( engine", "( air", "bags )", "( service", "hydraulic )", "( fuel"]:
    mm = re.search(rf"^\s{{2}}([A-Z/ ]+?)\s{{2,}}'{re.escape(term)}'\s+(\d+)/(\d+)\s*$", audit_before, re.M)
    assert mm, term
    leak_terms.append((term, mm.group(1).strip(), int(mm.group(2)), int(mm.group(3))))
resid = re.search(r"Texts still containing 'nhtsa campaign/recall' wording: ([\d,]+); recall-id pattern .*?: ([\d,]+); 'foia': ([\d,]+)", audit_after)
resid = [int(x.replace(",", "")) for x in resid.groups()]
paren_left = int(re.search(r"equal to a COMPDESC/top-level name: (\d+)", audit_after).group(1))

# ============================================================================ shared values
best = S("distilbert", "temporal")
mlp_t, bil_t = S("mlp", "temporal"), S("bilstm", "temporal")
per100 = lambda x: int(round(x * 100))
mc = {(r["model_a"], r["model_b"]): r for r in sig["mcnemar_exact_temporal"]}
pd_diff = sig["paired_difference_macro_f1_95ci"]
gap = {m: (S(m, "random").test_macro_f1 - S(m, "temporal").test_macro_f1) * 100 for m in MODELS}
gap_acc = {m: (S(m, "random").test_accuracy - S(m, "temporal").test_accuracy) * 100 for m in MODELS}
PN = PLAIN_NAMES
cls_plain = lambda c: PN[c]
per_day_2025 = PY[2025] / 365
growth_21_25 = (PY[2025] / PY[2021] - 1) * 100
assert 200 < per_day_2025 < 210 and per_day_2025 > 180 * 1.0


# ============================================================================ pages
def build_intro():
    s1 = [
        "Every car owner eventually meets a part that does not work the way it should.",
        "Most of the time it is an annoyance, such as a dead battery or a rattle behind the dashboard.",
        "Sometimes it is dangerous, such as brakes that feel soft, a steering wheel that locks up, or air bags that stay folded in a crash.",
        "In the United States, an owner who sees a problem like this can report it to the National Highway Traffic Safety Administration, known as NHTSA.",
        "The report is called a vehicle owner complaint, and owners can file one themselves or by phone.",
        "When they call, a staff member writes down what the owner says.",
        "NHTSA collects these complaints in a public file that anyone can download.",
        f"The file we used holds {n(facts['complaints_through_oct_2026'])} complaints received from January 2020 through October 2026.",
        f"That is an average of {n(facts['complaints_per_day_average'])} new complaints every day for almost seven years.",
    ]
    s2 = [
        "A complaint is a short piece of ordinary writing, and no two read alike.",
        f"The typical one in our collection is {n(facts['median_words'])} words long, about the length of a short paragraph.",
        "Some are a single sentence, like the owner whose air bags did not open when the car hit a concrete wall at 50 miles per hour.",
        "Others follow a problem for months, with dates, mileage and repeated trips to a dealer.",
        "Owners rarely use the names that engineers use for parts.",
        'They write that "the pedal becomes hard" or that "the steering wheel sticks" instead of naming a component.',
        f"About {n(round(facts['pct_transcripts_in_cleaned_pool']))} percent of the complaints we kept were written down by NHTSA staff during a phone call, in a more formal style, and the rest were written by the owners.",
        "Some owners write in anger, some in fear, and some with careful detail.",
        "All of them are trying to tell the government which part of the car failed.",
    ]
    s3 = [
        "The number of complaints has grown in recent years.",
        f"In 2021 NHTSA received {n(PY[2021])} complaints, and in 2025 it received {n(PY[2025])}.",
        f"That is about {n(round(growth_21_25))} percent more in four years.",
        f"The count for 2026 covers only January through October, and it already stands at {n(PY[2026])}.",
        "Each complaint is counted once, even when the owner lists several car parts.",
        "Behind every number is a person who noticed that something about a vehicle was unsafe or broken.",
        "The numbers also show why reading every complaint by hand is hard.",
        f"At the 2025 pace, NHTSA receives about {n(per_day_2025)} new complaints every day.",
        f"A person who read one complaint a minute would need more than {int(per_day_2025 // 60)} hours each day just to get through them.",
    ]
    s4 = [
        "A complaint helps only if the right people see it.",
        "Staff at NHTSA look for patterns, such as many owners of one model reporting the same failing part.",
        "When a pattern points to a safety defect, the agency can open an investigation, and a manufacturer can be asked to recall the vehicles for repair.",
        "To spot a pattern, someone first has to know which part each complaint is about.",
        "In the NHTSA file, every complaint is filed under a group of parts, such as the engine, the steering or the air bags.",
        "People assign those groups by reading the complaints.",
        "The job is slow, and two careful readers can disagree, because a story about a car that dies on the highway could point to the engine, the fuel system or the electrical system.",
        f"In fact, about {n(round(facts['pct_multi_label_dropped_after_merge']))} percent of the complaints in the file are filed under more than one group of parts, which shows how often one story touches several systems.",
        "This project asks a simple question: how well can the words an owner writes, and nothing else, tell us which part of the car failed?",
        "The rest of this website shows the data we used, the work we did to answer the question, and what we found.",
    ]
    tick = "".join(ticket(q) for q in quotes)
    head = phead("1 of 5 \u00b7 Introduction", "Why vehicle owners write to the government")
    td = tech("Technical details: where these numbers come from", [
        f"Source: NHTSA Office of Defects Investigation complaint files for 2020 to 2024 and 2025 to 2026. The files hold {n(funnel['raw_rows'])} rows for {n(funnel['raw_complaints'])} complaints, because a complaint that lists several components gets one row per component.",
        f"Counts per year: each complaint is counted once by the date NHTSA received it. The total of {n(facts['complaints_through_oct_2026'])} runs from 1 January 2020 to 31 October 2026 ({n(facts['days_covered'])} days).",
        f"Length: the cleaned collection of {n(facts['cleaned_pool_complaints'])} complaints has a median of {n(facts['median_words'])} words and a 95th percentile of {n(facts['p95_words'])} words.",
        f"Phone-call texts: {p1(facts['pct_transcripts_in_cleaned_pool'])}% of the cleaned complaints were written by NHTSA staff, recognized by their formal style (they start with &ldquo;the contact&rdquo; or a <code>TL*</code> marker).",
        f"Several groups: {n(facts['multi_label_dropped_after_merge'])} complaints ({p1(facts['pct_multi_label_dropped_after_merge'])}% of the file) map to more than one part group once two renamed labels are merged. Details are on the <a href=\"data.html\">Data &amp; Prep</a> page.",
    ])
    body = f"""<section>
<h2>When something goes wrong with a car</h2>
{para(s1, True)}
</section>
<section>
<h2>What a complaint looks like</h2>
{para(s2, True)}
<div class="tickets">{tick}</div>
<div class="small">Three real complaints with capital letters restored for reading. Each is one owner's report and says nothing about a brand. Source: NHTSA vehicle owner complaints, US government data, public domain.</div>
</section>
<section>
<h2>How many complaints there are</h2>
{para(s3, True)}
{pull(f"About {n(per_day_2025)} new complaints arrive every day.")}
{fig("fig_complaints_per_year.png", "Bar chart of vehicle complaints received by NHTSA each year: " + ", ".join(f"{y}: {n(c)}" for y, c in PY.items()) + ".", "Complaints received per year. The 2026 bar covers January through October. Data: results/eda/complaints_per_year.csv.")}
</section>
<section>
<h2>Why sorting complaints matters</h2>
{para(s4, True)}
{pull("How well can an owner's own words tell us which part failed?")}
{td}
</section>"""
    return page("introduction.html", "Introduction", body, "What vehicle owner complaints are, how many NHTSA receives, and why sorting them by failing part is hard.", head)


def build_conclusion():
    ek = overlap
    # the "hard for any reader" example: the both-wrong word-importance example whose true class is the electrical system
    ex_el = [e for e in wi["examples"] if e["category"] == "both_wrong" and e["true_label"] == "ELECTRICAL SYSTEM"]
    assert ex_el and ex_el[0]["pred_label"] == "FORWARD COLLISION AVOIDANCE", "conclusion example needs an electrical complaint sorted as collision avoidance"
    odi4 = ex_el[0]["odino"]
    n_fca = sum(int(P[m].loc[P[m].odino == odi4, "pred_label"].iloc[0] == "FORWARD COLLISION AVOIDANCE") for m in MODELS)
    NUMW = {0: "none of the three", 1: "one of the three", 2: "two of the three", 3: "all three"}
    mixd = {(r.true_class, r.put_under): (int(r["count"]), int(r.of_total)) for _, r in mix.iterrows()}
    fuel_eng, trans_eng = mixd[("FUEL/PROPULSION SYSTEM", "ENGINE")], mixd[("POWER TRAIN", "ENGINE")]
    el_eng, el_pt = mixd[("ELECTRICAL SYSTEM", "ENGINE")], mixd[("ELECTRICAL SYSTEM", "POWER TRAIN")]
    easiest = ident.sort_values(ascending=False)
    assert set(easiest.index[:2]) == {'Air bags', 'Steering'}
    hardest = ident.sort_values()
    c1 = [
        "We set out to learn whether the words an owner writes can tell us which part of the car failed.",
        f"To find out, we tested three computer methods on {n(best.n_test)} complaints received from May to October 2026.",
        "None of those complaints had been seen by the methods before, and all of them arrived after the complaints the methods learned from.",
        f"The best method put {per100(best.test_accuracy)} of every 100 of these complaints in the right one of ten groups.",
        f"The other two methods got {per100(mlp_t.test_accuracy)} and {per100(bil_t.test_accuracy)} right out of every 100.",
        "Careful checks show that the best method's lead over the other two is too large to be chance.",
        "The gap between the other two is small, and we cannot say one is better than the other.",
        "Some groups were easy and some were hard.",
        f"Air bags and steering complaints were the easiest, with {easiest.iloc[0]} of every 100 sorted correctly.",
        f"{hardest.index[0]} complaints were the hardest, at {hardest.iloc[0]} of every 100, {'with ' + hardest.index[1].lower() + ' level at ' + str(hardest.iloc[1]) if hardest.iloc[1] == hardest.iloc[0] else 'followed by ' + hardest.index[1].lower() + ' at ' + str(hardest.iloc[1])}.",
    ]
    c2 = [
        "The words that most clearly mark the easy groups are the part names themselves, such as \"air bag\" and \"steering wheel\".",
        "A complaint that says the air bags did not open leaves little doubt about where to file it.",
        "The hard groups are the ones where a symptom can have several causes.",
        "A car that stalls on the highway can have an engine problem, a fuel problem or an electrical problem, and the owner often cannot tell which.",
        "Our results show the same thing.",
        f"Of {fuel_eng[1]} fuel system complaints, {fuel_eng[0]} were put under the engine.",
        f"Of {trans_eng[1]} transmission complaints, {trans_eng[0]} were put under the engine as well.",
        f"Electrical complaints were spread across several other groups, with {el_eng[0]} of {el_eng[1]} going to the engine and {el_pt[0]} going to the transmission.",
        "These mistakes happen between groups whose problems can look alike in the owner's words.",
        "Reading the words alone works well for some problems and only fairly well for others.",
    ]
    c3 = [
        "Some complaints are hard for any reader.",
        f"Of the {n(ek['n_test'])} test complaints, {n(ek['wrong_for_all_three'])} ({p1(ek['pct_wrong_for_all_three'])} percent) were sorted wrongly by all three methods.",
        "One complaint describes a door sensor that fails now and then, and the car then brakes hard by itself at low speed.",
        f"NHTSA filed it under the electrical system, and {NUMW[n_fca]} sorted it as a collision-avoidance problem.",
        "Both answers are easy to defend, so the mistake says as much about the groups as about the computer.",
        "The groups themselves were chosen by NHTSA, and no independent check tells us they are always right.",
        f"We also left out the {n(facts['multi_label_dropped_after_merge'])} complaints ({n(round(facts['pct_multi_label_dropped_after_merge']))} percent of the file) that NHTSA filed under more than one group, so our results say nothing about those.",
        "We used ten groups of parts and balanced them so each group had a similar number of complaints, which is not how complaints arrive in real life.",
        "Our test covers one six-month window, from May to October 2026.",
        f"When we tested on a random mix of complaints instead of newer ones, the scores were between {p1(min(gap_acc.values()))} and {p1(max(gap_acc.values()))} points higher out of 100, so looking ahead in time costs only a little.",
    ]
    c4 = [
        "What does this mean for sorting real complaints?",
        f"A computer can read an owner's words and name the failing part about {per100(best.test_accuracy)} times out of 100 in our test.",
        "That is good enough to help a person sort a large pile of complaints into rough piles.",
        "It is not good enough to replace the person, especially for groups like the electrical system where about one in five is misfiled.",
        "The mistakes it makes are mostly between neighboring groups, such as the engine and the fuel system.",
        "A fair next step would be to let the computer sort first and have a person check the groups where mix-ups are common.",
        "Another would be to let it handle the complaints that list more than one part, which we skipped.",
        "We have shown that the owner's own words carry most of the answer.",
        "We have not shown that the methods would work on complaints from a much later year or from another country.",
        f'People who want to report a problem with their own vehicle can do so on <a href="{NHTSA_REPORT}">NHTSA\'s reporting page</a>.',
    ]
    head = phead("5 of 5 \u00b7 Conclusion", "What we found")
    mmm = mc[("mlp", "bilstm")]
    td = tech("Technical details: the numbers behind these statements", [
        f"Test set: complaints received 1 May to 31 October 2026, {n(best.n_test)} complaints with at most 800 per group. Each method was scored once with its best checkpoint on a separate validation set.",
        f"Scores on that set (accuracy, macro-F1): DistilBERT {f4(best.test_accuracy)} and {f4(best.test_macro_f1)}; MLP {f4(mlp_t.test_accuracy)} and {f4(mlp_t.test_macro_f1)}; BiLSTM {f4(bil_t.test_accuracy)} and {f4(bil_t.test_macro_f1)}.",
        f"DistilBERT against the other two: McNemar p-values {mc[('mlp', 'distilbert')]['p_value_exact']:.1g} (MLP) and {mc[('bilstm', 'distilbert')]['p_value_exact']:.1g} (BiLSTM); the 95% bootstrap intervals of the differences exclude zero. MLP against BiLSTM: p = {mmm['p_value_exact']:.3f}, which is not significant, with or without a correction for three comparisons.",
        f"Wrong for all three methods: {n(ek['wrong_for_all_three'])} of {n(ek['n_test'])} test complaints.",
        f"Random split instead of a time-based split: accuracy {gap_acc['mlp']:.2f}, {gap_acc['bilstm']:.2f} and {gap_acc['distilbert']:.2f} points higher for the MLP, BiLSTM and DistilBERT.",
        'Full tables, intervals, confusion matrices and error examples are on the <a href="results.html">Results</a> page.',
    ])
    body = f"""<section>
<h2>How well the computer sorted new complaints</h2>
{para(c1, True)}
{pull(f"{per100(best.test_accuracy)} of every 100 new complaints went to the right group.")}
{fig("fig_identified_correctly.png", "Bar chart of how many of every 100 complaints of each kind were sorted correctly: " + ", ".join(f"{k} {v}" for k, v in ident.sort_values(ascending=False).items()) + ".", "Out of every 100 complaints of each kind, how many the best method put in the right group. Data: results/figures/data/identified_correctly.csv.")}
</section>
<section>
<h2>Why some problems are harder</h2>
{para(c2, True)}
{fig("fig_mixups.png", "Bar charts of the most common mix-ups: fuel system complaints put under engine, transmission complaints put under engine, and electrical complaints put under other groups.", "The most common mix-ups on the test complaints. Red marks complaints put in the wrong group. Data: results/figures/data/mixups.csv.")}
</section>
<section>
<h2>Limits of what we found</h2>
{para(c3, True)}
</section>
<section>
<h2>What this could be used for</h2>
{para(c4, True)}
{td}
</section>"""
    return page("conclusion.html", "Conclusion", body, "What we found about sorting vehicle complaints by failing part, in plain language.", head)


def build_data():
    rl = rules
    ex = lambda k, i=0: rl[k]["examples"][i]
    pick = {1: 0, 2: 1, 3: 2, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 11: 1, 12: 1}
    desc = {1: "Remove reviewer tags at the end of a text (such as <code>*TR</code> or <code>*JB</code>) that NHTSA staff leave.",
            2: "Make the text lowercase.", 3: "Collapse runs of spaces into one space.",
            4: "Remove the <code>TL*</code> marker at the start of call-centre texts. The text is first flagged with <code>is_transcript</code>.",
            5: "Remove the sentence NHTSA adds when it hides personal details (FOIA redaction text), including cut-off versions.",
            6: "Remove sentences that mention the dealer &ldquo;VIN tool&rdquo;.",
            7: "Remove &ldquo;NHTSA campaign number: ID&rdquo; and the component name in parentheses after it. Leaves stubs such as &ldquo;notification of;&rdquo;.",
            8: "Remove any other parenthesized text that equals an NHTSA component name.",
            9: "Remove standalone recall IDs such as <code>19v757000</code>. The word &ldquo;recall&rdquo; stays.",
            10: "Collapse spaces again.",
            11: "Drop the whole complaint if it says the contact had not experienced a failure (recall-availability calls).",
            12: "Remove the template phrase &ldquo;parts distribution disconnect&rdquo; and keep the rest.", 13: "Collapse spaces again.",
            14: "Mask contact details typed into the text: phone numbers, e-mail addresses, VIN-like strings, street addresses and URLs become <code>[PHONE]</code>, <code>[EMAIL]</code>, <code>[VIN]</code>, <code>[ADDRESS]</code> and <code>[URL]</code>."}
    rows = []
    for k in sorted(rl):
        if k in pick and rl[k]["examples"]:
            b, a = ex(k, pick[k])
            exs = f"<code>{esc(b)}</code><br>&rarr; <code>{esc(a)}</code>"
        else:
            exs = "n/a"
        rows.append([k, desc[k], n(rl[k]["count"]), exs])
    rules_tbl = table(["#", "Rule (applied in this order)", "Texts changed", "Example, before and after"], rows,
                      caption="Cleaning rules. Counts are out of the complaints in the 10 classes. Source: results/cleaning_rules_log.txt.",
                      num_cols={0, 2}, text_cols={1, 3})
    class_rows = []
    for c in CLASSES:
        class_rows.append([f"<code>{esc(c)}</code>", esc(PN[c]), n(cb.loc[c, "cleaned_pool"])])
    class_tbl = table(["Label (NHTSA component group)", "Plain name", "Cleaned complaints"], class_rows,
                      caption="The 10 classes. Source: results/class_balance.csv.", num_cols={2})
    split_rows = [[esc(PN[c]), n(cb.loc[c, "temporal_train"]), n(cb.loc[c, "temporal_val"]), n(cb.loc[c, "temporal_test"]),
                   n(cb.loc[c, "random_train"]), n(cb.loc[c, "random_val"]), n(cb.loc[c, "random_test"])] for c in CLASSES]
    split_rows.append(["<strong>Total</strong>"] + [f"<strong>{n(cb[k].sum())}</strong>" for k in ["temporal_train", "temporal_val", "temporal_test", "random_train", "random_val", "random_test"]])
    split_tbl = table(["Class", "Temporal train", "Temporal val", "Temporal test", "Random train", "Random val", "Random test"], split_rows,
                      caption="Complaints per class in each split after the caps. Source: results/class_balance.csv.", num_cols={1, 2, 3, 4, 5, 6})
    leak_rows = [[f"<code>{esc(t)}</code>", esc(c), n(a), n(b), f"{a / b * 100:.1f}%"] for t, c, a, b in leak_terms]
    leak_tbl = table(["Term in the text", "Class where it appears most", "Texts in that class", "Texts with the term", "Share"], leak_rows,
                     caption="Giveaway terms found by the first leakage audit, before stripping. Source: results/leakage_audit_before_stripping.txt.", num_cols={2, 3, 4})
    field_rows = [["<code>ODINO</code>", "Complaint reference number. Repeats when a complaint lists several components.", "<code>odino</code>"],
                  ["<code>MAKETXT</code>, <code>MODELTXT</code>, <code>YEARTXT</code>", "Vehicle make, model and model year.", "<code>make</code>, <code>model</code>, <code>year</code> (shown, not used as model input)"],
                  ["<code>COMPDESC</code>", "NHTSA's component description, for example <code>SERVICE BRAKES, HYDRAULIC: FOUNDATION COMPONENTS</code>.", "<code>label</code> (the part before the first colon or comma)"],
                  ["<code>LDATE</code>", "Date NHTSA received the complaint (YYYYMMDD).", "<code>ldate</code>, used for the time-based split"],
                  ["<code>CDESCR</code>", "The complaint text.", "<code>text</code>, the only model input"],
                  ["(derived)", "True when the text was written down by NHTSA call-centre staff.", "<code>is_transcript</code>"],
                  ["(derived)", "train, val or test.", "<code>split</code>"]]
    field_tbl = table(["Raw field", "Meaning", "Published column"], field_rows, caption="Fields we keep. The full dictionary is on NHTSA's site.", text_cols={1, 2})
    t_ = {sp: int(cb[f"temporal_{sp}"].sum()) for sp in ["train", "val", "test"]}
    steps_html = '<ol class="steps">' + "".join(f'<li class="step{" is-key" if key else ""}"><span class="badge">{i}</span><h3>{ttl}</h3><div class="big">{big}</div><div class="what">{what}</div></li>' for i, (ttl, big, what, key) in enumerate([
        ("Raw file", n(funnel["raw_rows"]) + " rows", f"{n(funnel['raw_complaints'])} complaints, January 2020 to the start of November 2026", False),
        ("Label", f"{len(CLASSES)} classes", f"{n(funnel['single_label'])} complaints have one component group; {n(funnel['in_class_list'])} fall in the {len(CLASSES)} classes", False),
        ("Clean", n(funnel["cleaned_complaints"]), f"after stripping leaks and dropping {n(funnel['dropped_no_failure'])} no-failure calls, {n(funnel['dropped_duplicates'])} duplicates and {n(funnel['dropped_short'])} short texts", True),
        ("Split", n(sum(t_.values())), f"in the time-based sample: {n(t_['train'])} train, {n(t_['val'])} validation, {n(t_['test'])} test", False),
        ("Score", n(t_["test"]), "test complaints scored once per model", False)], 1)) + "</ol>"
    pipe_fig = fig("fig_pipeline.png", "Pipeline diagram: raw NHTSA file, cleaned dataset, splits, three models, evaluation, with the complaint counts at each step.", "The same five steps as a diagram, with the models and the checks. Counts: results/data_funnel.json and results/class_balance.csv.")
    try:
        ps = jl(R / "privacy_scan.json")["files"]["complaints_temporal.csv"]
    except Exception:
        ps = None
    if ps and (ps["phone_number"]["texts"] or ps["email_address"]["texts"] or ps["vin_like_17_chars"]["texts"]):
        priv_extra = (f" Some contact details typed into the complaint text itself remain in the published file: a phone number in {n(ps['phone_number']['texts'])} of {n(ps['rows'])} texts, "
                      f"an e-mail address in {n(ps['email_address']['texts'])} and a VIN-like string in {n(ps['vin_like_17_chars']['texts'])} (results/privacy_scan.json).")
    else:
        priv_extra = " Contact details typed into the text itself (phone numbers, e-mail addresses, VIN-like strings, street addresses and URLs) are replaced with <code>[PHONE]</code>, <code>[EMAIL]</code>, <code>[VIN]</code>, <code>[ADDRESS]</code> and <code>[URL]</code> by cleaning rule 14, and a scan of the published files finds none left (results/privacy_scan.json)."
        late = jl(R / "late_phone_masking.json")
        priv_extra += (f" {n(late['distinct_complaints_changed'])} texts had an additional phone format masked after training; the models saw those digits, and the effect on results is negligible. "
                       "None of them is in a test set (results/late_phone_masking.json).")
    privacy_callout = callout("Privacy", "<p>The raw file also holds the vehicle identification number (VIN), the owner's city, dealer names, phone numbers and addresses, and the vehicle operator's name. Our scripts never read those columns, and the published CSV files contain only the nine columns above. NHTSA already hides some personal details in the text with <code>[xxx]</code>." + priv_extra + "</p>", "plain")
    leak_callout = callout("Key finding", f"<p>The component name was sitting in the text. {n(rules[7]['count'])} complaints ({rules[7]['count'] / funnel['in_class_list'] * 100:.1f}%) carried it, written by NHTSA staff and not by the owner. A model reading it would not be reading the symptom.</p>", "error")
    d_train = (pd.Timestamp(VAL_START) - pd.Timestamp(facts["first_complaint_date"])).days
    d_val = (pd.Timestamp(TEST_START) - pd.Timestamp(VAL_START)).days
    d_test = (pd.Timestamp(TEST_END) - pd.Timestamp(TEST_START)).days + 1
    timeline_html = (f'<div class="timeline" role="img" aria-label="Time-based split: training before 1 November 2025, validation 1 November 2025 to 30 April 2026, test 1 May to 31 October 2026">'
                     f'<div class="tl-train" style="flex:{d_train}">Train</div><div class="tl-val" style="flex:{d_val}">Val</div><div class="tl-test" style="flex:{d_test}">Test</div></div>'
                     f'<div class="timeline-dates"><span>1 Jan 2020</span><span>1 Nov 2025</span><span>1 May 2026</span><span>31 Oct 2026</span></div>')
    body = f"""<section>
<h2>From file to dataset in five steps</h2>
{steps_html}
{pipe_fig}
</section>
<section>
<h2>Where the data comes from</h2>
<p>NHTSA's Office of Defects Investigation publishes every vehicle owner complaint as a tab-separated file with 51 fields per row. The files are on NHTSA's <a href="{NHTSA_DATA}">datasets page</a>, and the field dictionary is <a href="{NHTSA_DICT}">here</a>. We used the two most recent chunks, <a href="{NHTSA_FILES[0]}">COMPLAINTS_RECEIVED_2020-2024</a> and <a href="{NHTSA_FILES[1]}">COMPLAINTS_RECEIVED_2025-2026</a>. Together they hold {n(funnel['raw_rows'])} rows for {n(funnel['raw_complaints'])} complaints, received from January 2020 to the start of November 2026. There are more rows than complaints because a complaint that lists several components gets one row per component. The data is US government data and is public.</p>
<h3>The fields we keep</h3>
{field_tbl}
{privacy_callout}
</section>
<section>
<h2>Labels: which part failed</h2>
<p>The label is the first part of NHTSA's component description: the text before the first colon, then before the first comma. For example, <code>SERVICE BRAKES, HYDRAULIC: FOUNDATION COMPONENTS</code> becomes <code>SERVICE BRAKES</code>. We chose 10 groups with enough complaints to train and test on. All other labels are dropped, including <code>UNKNOWN OR OTHER</code>, <code>TIRES</code>, <code>SEATS</code> and <code>EXTERIOR LIGHTING</code>.</p>
{class_tbl}
<h3>Merged labels</h3>
<p>NHTSA renamed some groups over time, so two pairs of labels name the same parts. We merged <code>ENGINE AND ENGINE COOLING</code> into <code>ENGINE</code> and <code>FUEL SYSTEM</code> into <code>FUEL/PROPULSION SYSTEM</code> before any other step. We did not merge <code>VISIBILITY</code> into <code>VISIBILITY/WIPER</code>. Both labels were used through the whole period and they cover different parts (sunroof, windshield, glass and mirrors on one side, wipers on the other), so we dropped <code>VISIBILITY</code>.</p>
<h3>The multi-component rule</h3>
<p>{p1(facts['pct_complaints_with_multiple_component_rows'])}% of complaints ({n(facts['complaints_with_multiple_component_rows'])}) have rows for more than one component. After the merges above, {n(facts['multi_label_dropped_after_merge'])} complaints ({p1(facts['pct_multi_label_dropped_after_merge'])}%) map to more than one top-level label. We keep only complaints whose rows all map to exactly one label, which leaves {n(funnel['single_label'])}. Of those, {n(funnel['in_class_list'])} fall in the 10 classes. This keeps every label clean, and it also means our results say nothing about complaints that span several parts.</p>
</section>
<section>
<h2>Cleaning the text</h2>
<p>The cleaning is light on purpose. We lowercase the text, collapse spaces, and remove text that NHTSA staff or forms added, not the owner. The rules run in the order below. After them we drop exact duplicate texts ({n(funnel['dropped_duplicates'])}) and texts under 20 words ({n(funnel['dropped_short'])}).</p>
{rules_tbl}
<p>The funnel: {n(funnel['raw_rows'])} rows, {n(funnel['raw_complaints'])} complaints, {n(funnel['single_label'])} with one label, {n(funnel['in_class_list'])} in the 10 classes, minus {n(funnel['dropped_no_failure'])} complaints reporting no failure, {n(funnel['dropped_duplicates'])} duplicates and {n(funnel['dropped_short'])} short texts, leaves <strong>{n(funnel['cleaned_complaints'])}</strong> cleaned complaints.</p>
{fig("fig_cleaning_example.png", "One complaint before and after cleaning. The raw text is in capitals with a TL marker, a campaign number with a component name and reviewer tags highlighted as removed. The cleaned text is lowercase without them.", "One real complaint before and after cleaning. Data: results/figures/data/cleaning_example.json.")}
</section>
<section>
<h2>Leakage audit</h2>
<p>A model can score well for the wrong reason if the text contains the answer. We checked for this before training. For each class we listed the words and word pairs that most strongly point to it, and we looked for terms that appear almost only in one class.</p>
<p>The audit found a leak. NHTSA call-centre staff often write a sentence such as &ldquo;the contact received notification of NHTSA campaign number: 19V757000 (ENGINE AND ENGINE COOLING)&rdquo;. The component name in parentheses is the label itself. Rule 7 removed this phrase from {n(rules[7]['count'])} texts, which is {rules[7]['count'] / funnel['in_class_list'] * 100:.1f}% of the {n(funnel['in_class_list'])} complaints in the 10 classes. The table shows how clearly it showed up: a word pair such as <code>( air</code> appeared in {leak_terms[1][3]:,} texts, and {leak_terms[1][2]:,} of them were air bag complaints.</p>
{leak_callout}
{leak_tbl}
<p>We removed the phrase, any other parenthesized component name, and standalone recall IDs. We also removed two kinds of boilerplate that mark particular call types: the &ldquo;VIN tool&rdquo; sentences and the FOIA redaction text. A different problem was complaints that are not failure reports at all. Some call-centre texts say that the contact &ldquo;had not experienced a failure&rdquo; and only ask about a recall repair that is waiting for parts. Rule 11 dropped {n(rules[11]['count'])} of them.</p>
<p>After stripping, the audit finds {paren_left} texts with a parenthesized component name. It still finds {n(resid[0])} texts that mention an NHTSA campaign or recall in the owner's own words, {n(resid[1])} malformed recall IDs and {n(resid[2])} texts with a cut-off FOIA sentence. The full audit is in <a href="{gh('results/leakage_audit.txt')}">results/leakage_audit.txt</a>.</p>
<aside class="callout"><div class="callout-title">What we kept, and why it matters</div><div class="body"><ul>
<li><strong>Call-centre transcripts.</strong> {p1(facts['pct_transcripts_in_cleaned_pool'])}% of the cleaned complaints were written by NHTSA staff, in a formal style. We kept them and added the <code>is_transcript</code> column so we could score the two kinds of text separately (see <a href="results.html#transcripts">Results</a>).</li>
<li><strong>Brand and model names.</strong> Owners mention makes and models, and so do the models we trained. Some known problems belong to specific vehicles, so part of the score may come from recognizing a vehicle, not from reading the symptom. The scores may be lower for vehicles the models did not see.</li>
<li><strong>Stubs.</strong> Removing the campaign phrase leaves text such as &ldquo;the contact received notification of;&rdquo;. Some complaints have almost no information about the part left.</li>
</ul></div></aside>
</section>
<section>
<h2>Splits and sampling</h2>
<p>The main split is by time, because a real system would sort complaints that arrive after the ones it learned from. Training complaints were received before 1 November 2025, validation complaints from 1 November 2025 to 30 April 2026, and test complaints from 1 May to 31 October 2026. We left out the {n(funnel['raw_complaints'] - facts['complaints_through_oct_2026'])} complaints received on 1 November 2026, because that month was only one day long in the file. Each class is capped at 5,000 training complaints and 800 validation and 800 test complaints, chosen at random with seed 42. Classes with fewer complaints keep all of them: air bags, forward collision avoidance and windshield and wipers fall below the cap in validation and test.</p>
<p>The comparison split is a stratified random 80/10/10 split from the same cleaned pool, with the same caps. We used it for all three models to measure how much a random split overstates the score compared with the time-based split. In the random split, the train, validation and test sets are not separated by time, so complaints about the same recall can land on both sides.</p>
{timeline_html}
{split_tbl}
{fig("fig_class_balance.png", "Two bar charts. Left: cleaned complaints per class before sampling, from about 51,600 for engine to about 6,100 for forward collision avoidance. Right: complaints per class after sampling in the temporal split, mostly 6,600 per class.", "Class balance before and after sampling. Data: results/class_balance.csv.")}
<h3>Get the data and the code</h3>
<ul>
<li><a href="{gh('data/clean/complaints_temporal.csv')}">complaints_temporal.csv</a> (main split) and <a href="{gh('data/clean/complaints_random.csv')}">complaints_random.csv</a> (comparison split), with columns <code>odino, ldate, make, model, year, label, text, is_transcript, split</code>.</li>
<li>Scripts: <a href="{gh('src/01_download.py')}">01_download.py</a>, <a href="{gh('src/02_explore.py')}">02_explore.py</a>, <a href="{gh('src/03_clean_and_label.py')}">03_clean_and_label.py</a>, <a href="{gh('src/04_split_and_sample.py')}">04_split_and_sample.py</a>.</li>
<li>Raw source: <a href="{NHTSA_FILES[0]}">2020 to 2024</a>, <a href="{NHTSA_FILES[1]}">2025 to 2026</a>, and the <a href="{NHTSA_DICT}">field dictionary</a>.</li>
</ul>
</section>"""
    return page("data.html", "Data & Prep", body, "Where the NHTSA complaints come from, how they were labeled and cleaned, the leakage audit, and the train, validation and test splits.", phead("2 of 5 \u00b7 Data & Prep", "Data & Prep", "Where the complaints come from, what we removed because it gave the answer away, and how we split the data for training and testing."))


def hp_row(label, vals):
    return [label] + vals


def build_models():
    c = {m: cfg[(m, "temporal")] for m in MODELS}
    t = {m: tm[(m, "temporal")] for m in MODELS}
    rnd = {m: tm[(m, "random")] for m in MODELS}
    mlp_params = (t['mlp']['n_features'] * c['mlp']['hidden'][0] + c['mlp']['hidden'][0]) + (c['mlp']['hidden'][0] * c['mlp']['hidden'][1] + c['mlp']['hidden'][1]) + (c['mlp']['hidden'][1] * len(CLASSES) + len(CLASSES))
    hp_rows = [
        ["Input", "TF-IDF vector of words and 2-word phrases", f"first {c['bilstm']['max_len']} word tokens", f"first {c['distilbert']['max_len']} WordPiece tokens"],
        ["Features / vocabulary", f"{n(c['mlp']['max_features'])} features ({n(t['mlp']['n_features'])} used)", f"{n(t['bilstm']['vocab_size'])} words (min frequency {c['bilstm']['vocab_min_freq']}, cap {n(c['bilstm']['vocab_max'])})", f"{n(arch['vocab_size'])} (pre-trained)"],
        ["Layers", f"dense {c['mlp']['hidden'][0]}, dense {c['mlp']['hidden'][1]}, output {len(CLASSES)}", f"embedding {c['bilstm']['embed_dim']}, {c['bilstm']['num_layers']}-layer BiLSTM hidden {c['bilstm']['hidden']}, max+mean pooling, output {len(CLASSES)}", f"{arch['n_layers']} Transformer layers, {arch['n_heads']} heads, width {arch['dim']}, classifier head"],
        ["Parameters", n(mlp_params) + " (computed from the layer sizes)", n(t["bilstm"]["n_params"]), n(arch["n_params"])],
        ["Dropout", c["mlp"]["dropout"], c["bilstm"]["dropout"], f"{arch['classifier_dropout']} (head)"],
        ["Optimizer", "Adam", "Adam", "AdamW, weight decay " + str(c["distilbert"]["weight_decay"])],
        ["Learning rate", c["mlp"]["lr"], c["bilstm"]["lr"], f"{c['distilbert']['lr']} with {int(c['distilbert']['warmup_ratio'] * 100)}% linear warmup, then linear decay"],
        ["Batch size", c["mlp"]["batch_size"], c["bilstm"]["batch_size"], c["distilbert"]["batch_size"]],
        ["Gradient clipping", "none", c["bilstm"]["grad_clip"], 1.0],
        ["Maximum epochs", c["mlp"]["max_epochs"], c["bilstm"]["max_epochs"], c["distilbert"]["max_epochs"]],
        ["Early stopping", f"patience {c['mlp']['patience']}", f"patience {c['bilstm']['patience']}", "best of 3 epochs"],
        ["Best epoch / epochs run (temporal, random)", f"{t['mlp']['best_epoch']} of {t['mlp']['epochs_run']}, {rnd['mlp']['best_epoch']} of {rnd['mlp']['epochs_run']}",
         f"{t['bilstm']['best_epoch']} of {t['bilstm']['epochs_run']}, {rnd['bilstm']['best_epoch']} of {rnd['bilstm']['epochs_run']}", f"{t['distilbert']['best_epoch']} of {t['distilbert']['epochs_run']}, {rnd['distilbert']['best_epoch']} of {rnd['distilbert']['epochs_run']}"],
        ["Training time (temporal, random)", f"{t['mlp']['wall_time_s']:.0f} s, {rnd['mlp']['wall_time_s']:.0f} s", f"{t['bilstm']['wall_time_s'] / 60:.0f} min, {rnd['bilstm']['wall_time_s'] / 60:.0f} min", f"{t['distilbert']['wall_time_s'] / 60:.0f} min, {rnd['distilbert']['wall_time_s'] / 60:.0f} min"],
        ["Device", "MPS (Apple GPU)", "MPS (Apple GPU)", "MPS (Apple GPU)"],
    ]
    hp = table(["Setting", "MLP (TF-IDF)", "BiLSTM", "DistilBERT"], [[esc(str(x)) if i else esc(r[0]) for i, x in enumerate(r)] for r in hp_rows],
               caption="Settings and training facts. Source: results/&lt;model&gt;/&lt;split&gt;/config.json and metrics.json, results/distilbert/architecture.json.", text_cols={1, 2, 3})
    def mcard(key, title, role, how, why, link, alt, best=False):
        params = {"mlp": n(mlp_params), "bilstm": n(t["bilstm"]["n_params"]), "distilbert": n(arch["n_params"])}[key]
        r_ = S(key, "temporal")
        tt = t[key]["wall_time_s"]
        return (f'<article class="model-card{" best" if best else ""}"><div class="mc-head"><h3>{title}</h3>{pill(role, "amber" if best else "")}</div>'
                f'<div class="mc-diagram">{zoomimg(f"fig_arch_{key}.png", alt)}</div>'
                f'<dl><dt>Parameters</dt><dd>{params}</dd><dt>Training time</dt><dd>{(f"{tt:.0f} s" if tt < 120 else f"{tt / 60:.0f} min")}</dd>'
                f'<dt>Test macro-F1</dt><dd>{f4(r_.test_macro_f1)}</dd><dt>Test accuracy</dt><dd>{f4(r_.test_accuracy)}</dd></dl>'
                f'<div class="mc-text">{how}{why}{link}</div></article>')

    cards_html = mcard("mlp", "MLP", "baseline", f"""<p><strong>How it works.</strong> TF-IDF turns each complaint into a long list of numbers, one per word or two-word phrase. A word scores high when it is frequent in this complaint and rare across all complaints, so &ldquo;steering&rdquo; counts for more than &ldquo;the&rdquo;. The vocabulary is fit on the training texts only. A small network with two hidden layers learns which weighted words point to which class. It cannot see word order.</p>""", f"""<p><strong>Why we chose it.</strong> It is the baseline. It trains in about {t['mlp']['wall_time_s']:.0f} seconds, so it shows how far plain word counts go before anything more complex is needed.</p>""", f"""<p><a href="{gh('src/05_train_mlp.py')}">Training script</a></p>""", "Block diagram of the MLP: complaint text, TF-IDF features, two dense layers, predicted class.")
    cards_html += mcard("bilstm", "BiLSTM", "reads in order", f"""<p><strong>How it works.</strong> Each word becomes a vector of {c['bilstm']['embed_dim']} numbers that the model learns from scratch. Two LSTM layers then read the complaint from first word to last and from last to first, keeping a running memory of what they have seen. We take the largest and the average value of that memory over all word positions and feed both to a final layer. Padding is masked, so it cannot affect the pooling.</p>""", f"""<p><strong>Why we chose it.</strong> It is the classic way to read text in order. It can use phrases like &ldquo;will not start&rdquo; where the order carries meaning, which TF-IDF cannot. It trains from scratch, so it has no outside knowledge of English.</p>""", f"""<p><a href="{gh('src/06_train_bilstm.py')}">Training script</a></p>""", "Block diagram of the BiLSTM: tokenizer, embedding, two-layer BiLSTM, pooling, linear layer, predicted class.")
    cards_html += mcard("distilbert", "DistilBERT", "best score", f"""<p><strong>How it works.</strong> DistilBERT is a smaller version of BERT, a language model trained on a large collection of English text before we touched it. It has {arch['n_layers']} Transformer layers, and in each one every word looks at every other word to decide what it means in context. We add a small classifier on top and train the whole network on our complaints for up to {c['distilbert']['max_epochs']} epochs with a low learning rate. That step is called fine-tuning. It has {n(arch['n_params'])} parameters.</p>""", f"""<p><strong>Why we chose it.</strong> It brings general knowledge of English, which helps with owners who describe a symptom without naming the part. It is also small enough to train on a laptop in about {t['distilbert']['wall_time_s'] / 60:.0f} minutes. Texts longer than {c['distilbert']['max_len']} tokens are cut.</p>""", f"""<p><a href="{gh('src/07_train_distilbert.py')}">Training script</a></p>""", "Block diagram of DistilBERT: tokenizer, embeddings, six Transformer layers, first-token vector, classifier head, predicted class.", best=True)
    body = f"""<section>
<h2>What we predict, and why these three models</h2>
<p>The task is text classification. The input is one complaint text, and the output is one of the {len(CLASSES)} part groups from the <a href="data.html">Data &amp; Prep</a> page. The models see only the text. They do not get the make, model, year or date. We chose three models of increasing size so we could ask how much each step up buys: the MLP ignores word order, the BiLSTM reads in order, and DistilBERT starts from a language model trained on a large amount of general English.</p>
<p>All three share the same label encoding, splits and training rules. Each trains on the training set and stops early on validation macro-F1, the average of the per-class F1 scores. The best checkpoint on validation is kept, and the test set is scored once at the end, so nothing was tuned on test. The seed is 42 everywhere. Macro-F1 gives every class the same weight, which matters here because three classes are smaller than the others.</p>
{callout("Same rules for every model", "<p>Same text, same labels, same splits. Early stopping on validation macro-F1, and the test set scored once at the end. Nothing was tuned on test.</p>", "plain")}
</section>
<section>
<h2>Three models, side by side</h2>
<div class="section-lead">From a word-count baseline to a pre-trained language model. Click a diagram to enlarge it.</div>
<div class="model-cards">{cards_html}</div>
</section>
<section>
<h2>Settings</h2>
{hp}
<p>Shared code (data loading, label encoding, metrics, curves) is in <a href="{gh('src/common.py')}">common.py</a>. The evaluation script is <a href="{gh('src/08_evaluate.py')}">08_evaluate.py</a>. Training was done on a MacBook Air with an M4 chip and 16 GB of memory.</p>
</section>"""
    return page("models.html", "Models", body, "The three models we compared, how each works, why we chose them, and their settings.", phead("3 of 5 \u00b7 Models", "Models", "Three neural networks, from simple to large, trained on the same text, the same labels and the same splits."))


def render_wi(words, imps):
    mx = max(max(imps), -min(imps), 1e-9)
    out = []
    for w, v in zip(words, imps):
        r = abs(v) / mx
        if r < 0.08:
            out.append(f'<span title="{v:+.2f}">{esc(w)}</span>')
        else:
            out.append(f'<span class="{"sup" if v > 0 else "arg"}" style="--a:{0.2 + 0.7 * r:.2f}" title="{v:+.2f}">{esc(w)}</span>')
    return " ".join(out)


def snippets_html():
    out = ""
    names = {"both_correct": "both models right", "both_wrong": "both models wrong", "distilbert_right_bilstm_wrong": "right for DistilBERT, wrong for the BiLSTM"}
    for ex_i in range(1, 7):
        info = [e for e in wi["examples"] if e["example"] == ex_i]
        out += f"<h3>Example {ex_i}: {esc(PN[info[0]['true_label']])} ({names[info[0]['category']]})</h3>\n<div class=\"wi-pair\">\n"
        for e in sorted(info, key=lambda e: e["model"]):
            ok_ = e["pred_label"] == e["true_label"]
            head = (f'<strong>{NICE[e["model"]]}</strong> predicted {pill(PN[e["pred_label"]], "" if ok_ else "red")} '
                    f'<span class="mono">{e["confidence"] * 100:.0f}%</span> {pill("right" if ok_ else "wrong", "amber" if ok_ else "red")}')
            out += f'<div class="wi"><div class="wi-head">{head}</div><p class="wi-text">{render_wi(e["words"], e["importances"])}</p></div>\n'
        out += "</div>\n"
    return out


def scoreboard(split, title):
    import math
    lo_all = min(S(m, sp).test_macro_f1_ci_low for m in MODELS for sp in SPLITS)
    hi_all = max(S(m, sp).test_macro_f1_ci_high for m in MODELS for sp in SPLITS)
    a0, a1 = math.floor((lo_all - 0.002) * 200) / 200, math.ceil((hi_all + 0.002) * 200) / 200
    pos = lambda v: (v - a0) / (a1 - a0) * 100
    rows = ""
    for m in MODELS:
        r = S(m, split)
        rows += (f'<div class="sb-row"><div class="sb-name">{esc(NICE[m])}{pill("best score", "amber") if m == "distilbert" else ""}</div>'
                 f'<div class="sb-score">{f4(r.test_macro_f1)}<small>accuracy {f4(r.test_accuracy)}</small></div>'
                 f'<div class="sb-bar" role="img" aria-label="{esc(NICE[m])}: macro-F1 {f4(r.test_macro_f1)}, 95% interval {f4(r.test_macro_f1_ci_low)} to {f4(r.test_macro_f1_ci_high)}">'
                 f'<div class="sb-track"></div><div class="sb-ci c-{m}" style="left:{pos(r.test_macro_f1_ci_low):.2f}%;width:{pos(r.test_macro_f1_ci_high) - pos(r.test_macro_f1_ci_low):.2f}%"></div>'
                 f'<div class="sb-dot" style="left:{pos(r.test_macro_f1):.2f}%"></div>'
                 f'<div class="sb-ci-text">95% interval {f4(r.test_macro_f1_ci_low)} to {f4(r.test_macro_f1_ci_high)}</div></div></div>\n')
    ticks, v = "", math.ceil(a0 * 100) / 100
    while v <= a1 + 1e-9:
        ticks += f'<span style="left:{pos(v):.2f}%">{v:.2f}</span>'
        v += 0.01
    return f'<div class="scoreboard" role="group" aria-label="{esc(title)}"><h3 class="sb-title">{esc(title)}</h3>{rows}<div class="sb-axis">{ticks}</div></div>\n'


def build_results():
    srows, sbest = [], []
    for m in MODELS:
        for sc in SPLITS:
            r = S(m, sc)
            srows.append([NICE[m], sc, f4(r.val_accuracy), f4(r.val_macro_f1), f4(r.test_accuracy), f4(r.test_macro_f1),
                          f"{f4(r.test_macro_f1_ci_low)} to {f4(r.test_macro_f1_ci_high)}", f"{r.train_time_s:.0f} s" if r.train_time_s < 120 else f"{r.train_time_s / 60:.0f} min"])
    sbest = [i for i, r in enumerate(srows) if r[0] == "DistilBERT"]
    summ_tbl = table(["Model", "Split", "Val acc", "Val macro-F1", "Test acc", "Test macro-F1", "95% interval (test macro-F1)", "Train time"], srows,
                     caption="All models, both splits. Source: results/summary_table.csv. Test n = 7,320 (temporal) and 7,753 (random).", num_cols={2, 3, 4, 5, 7}, best_rows=sbest)
    mc_rows = [[f"{NICE[a]} vs {NICE[b]}", n(mc[(a, b)][f"only_{a}_correct"]), n(mc[(a, b)][f"only_{b}_correct"]), f"{mc[(a, b)]['p_value_exact']:.2g}"]
               for a, b in [("mlp", "bilstm"), ("mlp", "distilbert"), ("bilstm", "distilbert")]]
    mc_tbl = table(["Pair", "Only first is right", "Only second is right", "p-value (exact McNemar)"], mc_rows,
                   caption="McNemar's test on the temporal test set. Source: results/significance.json.", num_cols={1, 2, 3})
    diff_rows = []
    for sc in SPLITS:
        for k, v in pd_diff[sc].items():
            a, b = k.split(" minus ")
            diff_rows.append([f"{NICE[a]} minus {NICE[b]}", sc, f"{v['difference'] * 100:+.2f}", f"{v['ci_low'] * 100:+.2f} to {v['ci_high'] * 100:+.2f}", "yes" if v["ci_excludes_zero"] else "no"])
    diff_tbl = table(["Comparison", "Split", "Difference in test macro-F1 (points)", "95% interval (points)", "Interval excludes 0"], diff_rows,
                     caption="Paired bootstrap, 1,000 resamples of the test texts, seed 42. Source: results/significance.json.", num_cols={2})
    order = pcf1["distilbert_temporal"].sort_values(ascending=False).index
    pc_t = [[esc(PN[c]), f3(pcf1.loc[c, "mlp_temporal"]), f3(pcf1.loc[c, "bilstm_temporal"]), f3(pcf1.loc[c, "distilbert_temporal"])] for c in order]
    pc_r = [[esc(PN[c]), f3(pcf1.loc[c, "mlp_random"]), f3(pcf1.loc[c, "bilstm_random"]), f3(pcf1.loc[c, "distilbert_random"])] for c in order]
    pc_tbl = table(["Class", "MLP", "BiLSTM", "DistilBERT"], pc_t, caption="F1 score per class, temporal test set (main). Source: results/eval/per_class_f1_all.csv.", num_cols={1, 2, 3}) \
        + details("Same table for the random split", table(["Class", "MLP", "BiLSTM", "DistilBERT"], pc_r, caption="F1 score per class, random test set. Source: results/eval/per_class_f1_all.csv.", num_cols={1, 2, 3}))
    tr_rows = []
    for row in trans["rows"]:
        tr_rows.append([NICE[row["model"]], row["group"], n(row["n"]), f3(row["accuracy"]), f3(row["macro_f1"]), f"{f3(row['macro_f1_ci_low'])} to {f3(row['macro_f1_ci_high'])}", f3(row["macro_f1_classes_with_30plus_texts"])])
    tr_tbl = table(["Model", "Text type", "Texts", "Accuracy", "Macro-F1", "95% interval", "Macro-F1, classes with 30+ texts"], tr_rows,
                   caption="Temporal test set split by who wrote the text. Source: results/transcript_vs_owner.csv.", num_cols={2, 3, 4, 6})
    err = pd.read_csv(R / "eval" / "distilbert" / "temporal" / "error_examples.csv", dtype=str)
    err_rows = [[pill(PN[r.true_label]), pill(PN[r.pred_label], "red"), f"{float(r.confidence):.2f}", "call centre" if r.is_transcript == "True" else "owner", esc(r.text_first_300_chars)] for r in err.itertuples()]
    err_tbl = table(["True class", "Predicted class", "Confidence", "Written by", "Text (first 300 characters)"], err_rows,
                    caption="15 random misclassified test complaints from DistilBERT (temporal split, seed 42). Source: results/eval/distilbert/temporal/error_examples.csv.", num_cols={2}, text_cols={4})
    ok = pd.read_csv(R / "eval" / "distilbert" / "temporal" / "correct_examples.csv", dtype=str)
    ok_rows = [[esc(PN[r.true_label]), f"{float(r.confidence):.2f}", esc(r.text_first_300_chars)] for r in ok.itertuples()]
    ok_tbl = table(["Class (correct)", "Confidence", "Text (first 300 characters)"], ok_rows, caption="5 random correct predictions with confidence of 0.95 or more.", num_cols={1}, text_cols={2})
    cm_t = "".join(fig(f"confusion_{m}_temporal_normalized.png", f"Normalized confusion matrix for {NICE[m]} on the temporal test set. Rows are true classes, columns are predicted classes.", f"{NICE[m]}, temporal test set, rows sum to 1.") for m in MODELS)
    cm_c = "".join(fig(f"confusion_{m}_temporal_counts.png", f"Confusion matrix of counts for {NICE[m]} on the temporal test set.", f"{NICE[m]}, temporal test set, counts.") for m in MODELS)
    cm_r = "".join(fig(f"confusion_{m}_random_{k}.png", f"Confusion matrix ({k}) for {NICE[m]} on the random test set.", f"{NICE[m]}, random test set, {k}.") for m in MODELS for k in ["normalized", "counts"])
    cu_t = "".join(fig(f"curves_{m}_temporal.png", f"Training curves for {NICE[m]} on the temporal split: loss and validation macro-F1 per epoch.", f"{NICE[m]}, temporal split.") for m in MODELS)
    cu_r = "".join(fig(f"curves_{m}_random.png", f"Training curves for {NICE[m]} on the random split.", f"{NICE[m]}, random split.") for m in MODELS)
    ov = overlap
    top = lambda m: em[(m, "temporal")]["top_confused_pairs"]
    pair_txt = lambda x: f"{' versus '.join(PN[c].lower() for c in x['pair'].split(' / '))} ({x['count']} complaints)"
    # second-lowest class by F1 for each model (temporal) for the sentence after the electrical-system finding
    second = {m: pcf1[f"{m}_temporal"].sort_values().index[1] for m in MODELS}
    assert len(set(second.values())) == 1, second
    next_weak = PN[next(iter(second.values()))].lower()
    # keyword example: the both-wrong electrical complaint; the quoted words must be top supporting words for both models
    kw = [e for e in wi["examples"] if e["category"] == "both_wrong" and e["true_label"] == "ELECTRICAL SYSTEM"]
    assert {e["model"] for e in kw} == {"bilstm", "distilbert"}
    for e in kw:
        assert "braking" in [x["word"] for x in e["top_supporting_words"]], e["top_supporting_words"]
    kw_ex = kw[0]["example"]
    d_el = pcf1.loc["ELECTRICAL SYSTEM"]
    sa = stub["accuracy"]["distilbert"]
    t_bil_trunc = tm[("bilstm", "temporal")]["pct_truncated"]
    mc_md = mc[("mlp", "distilbert")]
    mc_mb = mc[("mlp", "bilstm")]
    sb_t = scoreboard("temporal", "Time-based test (main): complaints from May to October 2026")
    sb_r = scoreboard("random", "Random split test (comparison)")
    cm_figs = "".join(fig(f"confusion_{m}_{sc}_{k}.png", f"{'Share of each true class' if k == 'normalized' else 'Counts'} confusion matrix for {NICE[m]} on the {sc} test set. Rows are true classes, columns are predicted classes.",
                          f"{NICE[m]}, {sc} test set, {'share of each true class' if k == 'normalized' else 'counts'}.", f' data-model="{m}" data-split="{sc}" data-kind="{k}"')
                      for m in MODELS for sc in SPLITS for k in ["normalized", "counts"])
    cm_viewer = f'<div class="cm-viewer"><div class="figrow">{cm_figs}</div></div>'
    err_callout = callout("Key finding", f"<p>{n(ov['wrong_for_all_three'])} of {n(ov['n_test'])} test complaints ({p1(ov['pct_wrong_for_all_three'])}%) are wrong for all three models, so these mistakes are not one model's quirk.</p>", "error")
    body = f"""<section id="summary">
<h2>Summary</h2>
<p>All numbers are on the test set, which each model scored once with its best validation checkpoint. <strong>Accuracy</strong> is the share of test complaints sorted correctly. <strong>Macro-F1</strong> averages the F1 score of the {len(CLASSES)} classes, so a small class counts as much as a large one. The interval is a 95% bootstrap interval from 1,000 resamples of the test texts.</p>
{sb_t}{sb_r}
{details("Full table: validation and test, both splits", summ_tbl)}
<p>On the time-based test (complaints from May to October 2026), DistilBERT reaches {f4(best.test_accuracy)} accuracy and {f4(best.test_macro_f1)} macro-F1. The MLP reaches {f4(mlp_t.test_macro_f1)} and the BiLSTM {f4(bil_t.test_macro_f1)}. The test classes are capped at 800, so the class mix is balanced and differs from the real mix of complaints.</p>
</section>
<section id="significance">
<h2>Are the differences real?</h2>
<p>We checked two ways. McNemar's test compares the models on the same test complaints: it counts the complaints where one model is right and the other is wrong. The paired bootstrap resamples the test set and recomputes the difference in macro-F1.</p>
{mc_tbl}
{diff_tbl}
<p>DistilBERT beats both other models on both splits: the intervals exclude 0, and the McNemar p-values on the temporal set are {mc_md['p_value_exact']:.1g} (against the MLP) and {mc[('bilstm', 'distilbert')]['p_value_exact']:.1g} (against the BiLSTM). The MLP and the BiLSTM are a weak case. On the temporal split the BiLSTM is {abs(pd_diff['temporal']['bilstm minus mlp']['difference']) * 100:.1f} points lower, the interval includes 0 and the McNemar p-value is {mc_mb['p_value_exact']:.3f}, which is above 0.05 and would also fail a correction for three comparisons (the cutoff is 0.0167). On the random split the interval includes 0 as well. We treat the two as roughly equal, with the MLP slightly ahead on the temporal split.</p>
</section>
<section id="split">
<h2>Time-based split vs random split</h2>
{fig("fig_split_comparison.png", "Dot chart of test macro-F1 for each model on the temporal and random splits with 95% intervals. The random split scores higher for every model.", "Test macro-F1 with 95% bootstrap intervals. The axis does not start at zero. Data: results/summary_table.csv.")}
<p>A random split scores higher for every model: {gap['mlp']:.2f} points for the MLP, {gap['bilstm']:.2f} for the BiLSTM and {gap['distilbert']:.2f} for DistilBERT in macro-F1. The time-based split is the fairer test of a system that sorts future complaints, so we report it as the main result. The gap is small, and we did not study why.</p>
</section>
<section id="per-class">
<h2>Results by class</h2>
{fig("fig_per_class_f1.png", "Dot chart of per-class F1 on the temporal test set for the three models. Steering and air bags are highest, electrical system is lowest for every model.", "F1 by class on the temporal test set. Data: results/eval/per_class_f1_all.csv.")}
{pc_tbl}
<p>Steering and air bags are easiest. Electrical system is the weakest class for every model on both splits: F1 {f3(d_el['mlp_temporal'])} for the MLP, {f3(d_el['bilstm_temporal'])} for the BiLSTM and {f3(d_el['distilbert_temporal'])} for DistilBERT on the temporal test. DistilBERT improves it by {(d_el['distilbert_temporal'] - d_el['mlp_temporal']) * 100:.1f} points over the MLP, but it stays the weakest. The {next_weak} is next-lowest for every model.</p>
</section>
<section id="confusion">
<h2>Confusion matrices</h2>
<p>Each row is the true class and each column is the predicted class. A perfect model would have 1.00 on the diagonal. Amber cells are correct answers and red cells are mix-ups. With JavaScript on, the tabs show one model, split and view at a time; without it all twelve matrices are listed.</p>
{cm_viewer}
<p>The top mix-ups on the temporal test, both directions combined, are in the table below. Source: results/eval/&lt;model&gt;/temporal/metrics.json.</p>
{table(["Model", "Pair 1", "Pair 2", "Pair 3"], [[NICE[m]] + [f"{esc(' / '.join(PN[x] for x in top(m)[i]['pair'].split(' / ')))}: {top(m)[i]['count']}" for i in range(3)] for m in MODELS], text_cols={1, 2, 3})}
</section>
<section id="curves">
<h2>Training curves</h2>
<p>The left panel shows training and validation loss, the right panel shows validation macro-F1. The ringed point marks the checkpoint we kept. The MLP overfits after the first epoch, which is why its best epoch is {tm[('mlp', 'temporal')]['best_epoch']}.</p>
<div class="figrow three">{cu_t}</div>
{details("Random split curves", f'<div class="figrow three">{cu_r}</div>')}
</section>
<section id="transcripts">
<h2>Call-centre texts vs owner-written texts</h2>
<p>{n(trans['rows'][0]['n'] + trans['rows'][3]['n'])} test complaints split into {n(trans['rows'][0]['n'])} owner-written texts and {n(trans['rows'][3]['n'])} call-centre texts. Accuracy is about the same for both. Macro-F1 looks lower for call-centre texts, but only {trans['class_support_by_group']['transcript']['VISIBILITY/WIPER']} of them belong to the windshield and wiper class, so that class's F1 is very noisy. The last column drops classes with fewer than 30 texts in the group, and then the two groups score in the same range for every model. We found no sign that the formal call-centre style makes the task easier.</p>
{tr_tbl}
</section>
<section id="errors">
<h2>Where the models fail</h2>
{err_callout}
<ul>
<li><strong>Neighboring groups.</strong> For DistilBERT the largest mix-up is {pair_txt(top('distilbert')[0])}, followed by {pair_txt(top('distilbert')[1])} and {pair_txt(top('distilbert')[2])}. A symptom such as a car that stalls, shudders or loses power can fit several systems, and a sunroof that breaks can count as body or as glass.</li>
<li><strong>Shared mistakes.</strong> {n(ov['wrong_for_at_least_one'])} of {n(ov['n_test'])} test complaints are wrong for at least one of the three models, and the {n(ov['wrong_for_all_three'])} in the box above are wrong for all of them.</li>
<li><strong>Stubs.</strong> Some call-centre texts keep only a stub such as &ldquo;the contact received notification of; however, the part to do the recall repair was not yet available&rdquo;, because we removed the campaign number and component name. {n(stub['n_with_stub'])} test texts contain such a stub. DistilBERT is right on {p1(sa['with_stub'] * 100)}% of them, against {p1(sa['without_stub'] * 100)}% of the other texts.</li>
<li><strong>Keywords.</strong> The highlights below show a model leaning on one word. In example {kw_ex}, the word &ldquo;braking&rdquo; is among the top supporting words for both models and pulls a complaint about a door sensor toward collision avoidance, even though NHTSA filed it under the electrical system.</li>
<li><strong>Labels are not ground truth.</strong> NHTSA staff assign the groups, and a complaint that fits two groups gets only one label here.</li>
<li><strong>Truncation.</strong> The BiLSTM keeps the first {cfg[('bilstm', 'temporal')]['max_len']} tokens: {p1(t_bil_trunc['train'] * 100)}% of training texts and {p1(t_bil_trunc['test'] * 100)}% of test texts are longer.</li>
<li><strong>Vehicle names.</strong> Brand and model names stay in the text. Some of the score may come from knowing which vehicles have which known problems.</li>
<li><strong>Balanced test.</strong> Classes are capped at 800 in the test set, so the numbers describe a balanced mix, not the real mix of complaints. Windshield and wipers, air bags and collision avoidance have fewer than 800 test complaints.</li>
<li><strong>Scope.</strong> We use only complaints with one label, in ten groups, from one six-month test window.</li>
</ul>
<h3>Examples of mistakes</h3>
{details("15 random misclassified complaints (DistilBERT, temporal)", err_tbl + f'<p class="small">Other models: <a href="{gh("results/eval/mlp/temporal/error_examples.csv")}">MLP</a>, <a href="{gh("results/eval/bilstm/temporal/error_examples.csv")}">BiLSTM</a>. Complaints wrong for all three models: <a href="{gh("results/eval/errors_all_models.csv")}">errors_all_models.csv</a>.</p>')}
{details("5 random confident correct predictions", ok_tbl)}
</section>
<section id="highlights">
<h2>Which words drive a prediction</h2>
<p>For six test complaints we removed one word at a time and measured how much the model's confidence in its own answer dropped. Amber words support the predicted class and gray words argue against it. The value is the drop in the log-odds of the predicted class (hover over a word to see it). Two examples were right for both models, two were wrong for both, and two were right for DistilBERT and wrong for the BiLSTM. We drew them at random from complaints of 25 to 90 words (seed 42), two from each case. Method and values: results/eval/word_importance/word_importance.json.</p>
<div class="legend"><span><span class="swatch" style="background:rgba(242,169,0,.7)"></span>supports the predicted class</span><span><span class="swatch" style="background:rgba(92,102,116,.55)"></span>argues against it</span></div>
{snippets_html()}
<p class="small">Removing one word at a time understates words that are repeated or have close substitutes, so a confident model can show small values everywhere. These examples illustrate the models. They do not prove how the models work.</p>
</section>
<section>
<h2>Files</h2>
<p>Everything is in the repository: <a href="{gh('results/summary_table.csv')}">summary_table.csv</a>, <a href="{gh('results/significance.json')}">significance.json</a>, <a href="{gh('results/transcript_vs_owner.csv')}">transcript_vs_owner.csv</a>, and the per-model folders under <a href="{REPO}/tree/main/results">results/</a>. The evaluation script is <a href="{gh('src/08_evaluate.py')}">08_evaluate.py</a>.</p>
</section>"""
    return page("results.html", "Results", body, "Test results for the MLP, BiLSTM and DistilBERT on both splits, with significance tests, confusion matrices, error analysis and word highlights.", phead("4 of 5 \u00b7 Results", "Results", "DistilBERT is the best model on both splits. The MLP and the BiLSTM are close to each other. Electrical system complaints are the hardest class for every model."))


def explore_home():
    if not EXPLORE:
        return ""
    cards = "".join(f'<li><a href="{f}"><span class="ttl">{esc(t)}</span><span class="what">{esc(d)}</span></a></li>' for f, t, d in EXPLORE)
    return (f'<section id="explore"><h2>Explore</h2><div class="section-lead">Hands-on pages that sit beside the story. They use the same data and the same results.</div>'
            f'<ol class="how explore-cards">{cards}</ol></section>')


def build_index():
    st = "".join([stat(n(funnel["cleaned_complaints"]), "complaints analyzed after cleaning"),
                  stat(str(len(CLASSES)), "vehicle systems the computer has to tell apart"),
                  stat(f"{per100(best.test_accuracy)} of 100", "new complaints put in the right system by the best model"),
                  stat(n(overlap["wrong_for_all_three"]), f"test complaints all three models got wrong ({p1(overlap['pct_wrong_for_all_three'])}%)", True)])
    head = f"""<div class="hero"><div class="inner">
<div class="kicker">A data story about vehicle safety complaints</div>
<h1>What Failed on This Car?</h1>
<div class="subtitle">When a car part fails, owners can report it to the US government in their own words, and this project asks whether a computer can read those words and name the part that failed.</div>
<div class="stats">{st}</div>
</div></div>
"""
    steps = [("introduction.html", "Introduction", "What owner complaints are, how many there are, and why sorting them is hard."),
             ("data.html", "Data", "The NHTSA files, the labels, the cleaning, and the leaks we found and removed."),
             ("models.html", "Models", "Three neural networks, from a word-count baseline to DistilBERT."),
             ("results.html", "Results", "Scores with intervals, tests, mistakes, and which words drive a prediction."),
             ("conclusion.html", "Conclusion", "What we found, in plain language.")]
    how = "".join(f'<li><a href="{h}"><span class="num">{i}</span><span class="ttl">{esc(t)}</span><span class="what">{esc(d)}</span></a></li>' for i, (h, t, d) in enumerate(steps, 1))
    body = f"""<section>
<h2>How to read this site</h2>
<div class="section-lead">Five parts, in the order we worked. Each page ends with a link to the next one.</div>
<ol class="how">{how}</ol>
{callout("Short on time?", '<p>Read the Introduction and the Conclusion. They use plain language and no jargon. The pages in between hold the data, the models and the numbers behind every statement.</p>', "plain")}
{fig("fig_identified_correctly.png", "Bar chart of how many of every 100 complaints of each kind were sorted correctly, from air bags and steering at 96 to electrical system at 81.", "Out of every 100 complaints of each kind, how many the best method put in the right group (complaints from May to October 2026).")}
<div class="small">A class project using public NHTSA complaint data. Not an NHTSA product, and not safety advice.</div>
</section>
{explore_home()}"""
    return page("index.html", "Home", body, "A class project that sorts NHTSA vehicle owner complaints by the failing part using three neural network models.", head)


def build_about():
    body = f"""<section>
<h2>The team</h2>
<ul>
<li>Tejas Naik</li>
<li>[Teammate 2]</li>
<li>[Teammate 3]</li>
</ul>
<p>Course: [Course name], [Semester].</p>
<p>Code, cleaned data and all results: <a href="{REPO}">{REPO.replace("https://", "")}</a>.</p>
</section>
<section>
<h2>The project</h2>
<p>This is an open team project for a deep learning class. The question is whether the words of a vehicle owner complaint are enough to name the vehicle part that failed. We compared three neural networks on {n(funnel['cleaned_complaints'])} cleaned complaints from NHTSA and report where they work and where they do not.</p>
{fig("fig_pipeline.png", "Pipeline diagram from the raw NHTSA file to the evaluation of three models.", "The project in five steps.")}
</section>
<section>
<h2>How this project was made</h2>
{callout("AI assistance", "<p>The code, analysis pipeline, figures, and website text were developed with Claude Code (Anthropic). The team directed the project, made the design decisions (class list, merges, leakage rules, splits, model choices), and reviewed the results.</p>", "")}
</section>
<section>
<h2>Data</h2>
<p>The complaints come from the National Highway Traffic Safety Administration (NHTSA), Office of Defects Investigation. NHTSA publishes them on its <a href="{NHTSA_DATA}">datasets page</a>, and the data is US government data in the public domain. This project is not made by, endorsed by or connected to NHTSA. We published a cleaned sample with only the columns <code>odino, ldate, make, model, year, label, text, is_transcript, split</code>, left out all personal fields, and replaced contact details inside the text with placeholders.</p>
</section>
<section>
<h2>Tools</h2>
<ul>
<li>Python {env['python']}, PyTorch {env['torch']} (Apple MPS backend), Hugging Face transformers {env['transformers']}, scikit-learn {env['scikit-learn']}, pandas {env['pandas']}, NumPy {env['numpy']}, SciPy {env['scipy']}, matplotlib {env['matplotlib']} and seaborn {env['seaborn']}. Pinned versions are in <a href="{gh('requirements.txt')}">requirements.txt</a>.</li>
<li>Hardware: a MacBook Air with an M4 chip and 16 GB of memory. No cloud services were used.</li>
<li>The website is plain HTML, one CSS file and one small JavaScript file for the figure viewer and the matrix tabs (every page works without it). Headings use Barlow Condensed, body text Inter and numbers JetBrains Mono, loaded from Google Fonts; the figures use the same open-licensed fonts, kept in <code>assets/fonts/</code>. It is built from the results files by <a href="{gh('src/10_build_site.py')}">10_build_site.py</a>, so the numbers on each page come from <code>results/</code>.</li>
<li>Random seed 42 is used everywhere. The code is released under the MIT license.</li>
</ul>
</section>
<section>
<h2>Please note</h2>
{callout("Not a safety assessment", f"<p>These results describe one dataset and one test window. They are not a safety assessment of any vehicle.</p><p>To report a safety problem with a vehicle, use <a href=\"{NHTSA_REPORT}\">NHTSA's reporting page</a>.</p>", "")}
</section>"""
    return page("about.html", "About", body, "About this class project: the team, the question, the data source, how it was made and the tools.", phead("About", "About", "A class project in deep learning, with all code, data and results published."))


# ============================================================================ explore pages
def mb(x):
    return f"{x / 1e6:.1f} MB"


def build_try():
    chk = jl(R / "browser_model_check.json")
    mlp_row, dil_row = S("mlp", "temporal"), S("distilbert", "temporal")
    meta_bytes = (DOCS / "data" / "mlp_vocab.json").stat().st_size
    w_bytes = (DOCS / "data" / "mlp_weights.bin").stat().st_size
    smp_bytes = (DOCS / "data" / "sample_complaints.json").stat().st_size
    smp = jl(DOCS / "data" / "sample_complaints.json")
    n_smp, per_cls = len(smp["items"]), len(smp["items"]) // len(CLASSES)
    head = phead("Explore", "Try it yourself", "Type a complaint, or load a real one, and watch our simplest model sort it into one of the ten systems. It runs inside your browser.")
    body = f"""<section>
{callout("Which model is this?", f"<p>This is the MLP (TF-IDF), our simplest model. On the time-based test it has a macro-F1 of {f4(mlp_row.test_macro_f1)} and an accuracy of {f4(mlp_row.test_accuracy)}. DistilBERT, our best model, scores {f4(dil_row.test_macro_f1)} and {f4(dil_row.test_accuracy)}, but it has too many weights to download into a web page, so it is not here. The scores come from the <a href=\"results.html\">Results</a> page.</p>", "plain")}
<noscript><div class="callout error"><div class="callout-title">JavaScript is off</div><div class="body"><p>This page runs the model in your browser, so it needs JavaScript. Without it you can still read how it works: the page loads the TF-IDF vocabulary and the weights of the MLP, turns your text into word-count scores, and passes them through three layers to get one score per system. The <a href="models.html">Models</a> page explains the same steps, and the <a href="results.html">Results</a> page shows how the model did on real test complaints.</p></div></div></noscript>
<div class="tool" id="try-app" hidden>
<div class="loadbar" id="try-load"><div class="loadtext" id="try-load-text">Loading the model: 0.0 of {mb(meta_bytes + w_bytes)}</div><progress id="try-progress" max="100" value="0" aria-label="Model download progress"></progress><div class="small">Download size: {mb(meta_bytes)} of vocabulary and {mb(w_bytes)} of weights (about {mb(smp_bytes)} more if you load a real complaint). Nothing you type leaves your device.</div></div>
<label class="field-label" for="try-text">Complaint text</label>
<textarea id="try-text" rows="7" maxlength="4000" placeholder="Describe what failed, in your own words. For example: the steering wheel locked up while I was driving." disabled></textarea>
<div class="btns"><button class="btn primary" id="try-go" type="button" disabled>Predict</button><button class="btn" id="try-real" type="button" disabled>Load a real complaint</button><button class="btn quiet" id="try-clear" type="button" disabled>Clear</button></div>
<div id="try-out" aria-live="polite"></div>
<div class="small" id="try-source" hidden></div>
</div>
</section>
<section>
<h2>How to read the result</h2>
<ul>
<li><strong>The bars</strong> show the three systems the model finds most likely. The percentages are the model's own confidence and they add up to 100% across all ten systems. A high number is not a promise: the model can be confident and wrong.</li>
<li><strong>The highlighted words</strong> come from removing one word at a time and checking how the confidence in the top answer changes. Amber words push toward the top answer. Gray words push away from it.</li>
<li><strong>Real complaints</strong> come from a sample of {n(n_smp)} complaints in our time-based test set, {per_cls} per system. They are the masked text from the published data, and the model never saw them in training. After you predict, the page shows where NHTSA filed the complaint.</li>
</ul>
</section>
<section>
<h2>Is the browser copy the same as the trained model?</h2>
<p>We rewrote the word-count scoring (TF-IDF with single words and two-word phrases) and the three layers in JavaScript. To keep the download small, the weights are stored as 8-bit integers with one scale per row. We then ran the original Python model and this browser code on the same {n(chk['n_texts'])} test complaints. They picked the same system for {chk['agreement_pct']:g}% of them, and the largest difference in any probability was {chk['max_abs_probability_difference']:.4f}. The report is in <a href="{gh('results/browser_model_check.json')}">results/browser_model_check.json</a>.</p>
{callout("Not a safety tool", "<p>The model sees only the words you give it. It was trained on one source of complaints, and it knows nothing about your vehicle. Do not use it to decide whether a car is safe. To report a problem, use <a href=\"" + NHTSA_REPORT + "\">NHTSA's reporting page</a>.</p>", "plain")}
</section>"""
    return page("try.html", "Try it", body, "Run our trained MLP in your browser: type a vehicle complaint and see which of ten systems it is sorted into.", head, scripts=["mlp.js", "try.js"])


# ============================================================================ write and check
BANNED = ["revolutionary", "cutting-edge", "robust", "leverage", "delve", "seamless", "crucial", "powerful", "state-of-the-art", "game-chang",
          "unlock", "harness", "comprehensive", "novel", "groundbreaking", "innovative", "pivotal", "journey", "deep dive", "dive into", "landscape", "tapestry"]


def text_of(h):
    h = re.sub(r"(?s)<(script|style).*?</\1>", "", h)
    return html.unescape(re.sub(r"<[^>]+>", " ", h))


def check(pages):
    problems = []
    for fname, h in pages.items():
        t = text_of(h)
        for ch in ["—", "–"]:
            if ch in t or ch in h:
                problems.append(f"{fname}: contains dash character {ch!r}")
        low = t.lower()
        for w in BANNED:
            if w in low:
                problems.append(f"{fname}: banned word {w!r}")
        for src in re.findall(r'src="(img/[^"]+)"', h):
            if not (DOCS / src).exists():
                problems.append(f"{fname}: missing image {src}")
        for src in re.findall(r'<script src="(js/[^"]+)"', h):
            if not (DOCS / src).exists():
                problems.append(f"{fname}: missing script {src}")
        for href in re.findall(r'href="([^"#:]+\.(?:html|css))', h):
            if not (DOCS / href).exists():
                problems.append(f"{fname}: broken link {href}")
        if "<h1" not in h or h.count("<h1") != 1:
            problems.append(f"{fname}: needs exactly one h1")
        if 'alt=""' in h:
            problems.append(f"{fname}: empty alt text")
    for fname in ["introduction.html", "conclusion.html"]:
        low = text_of(re.sub(r"(?s)<details class=tech>.*?</details>", "", pages[fname]))
        for term in ["MLP", "BiLSTM", "DistilBERT", "F1", "confusion matrix", "macro", "TF-IDF", "neural network"]:
            if term.lower() in low.lower():
                problems.append(f"{fname}: technical term {term!r}")
    return problems


def main():
    pages = {"index.html": build_index(), "introduction.html": build_intro(), "data.html": build_data(), "models.html": build_models(),
             "results.html": build_results(), "conclusion.html": build_conclusion(), "about.html": build_about()}
    builders = {"try.html": build_try}
    for f in EXPLORE_FILES:
        pages[f] = builders[f]()
    for fname, h in pages.items():
        (DOCS / fname).write_text(h)
    (DOCS / ".nojekyll").write_text("")
    probs = check(pages)
    print("pages written:", ", ".join(pages))
    print("problems:", len(probs))
    for p in probs:
        print("  -", p)
    for fname in ["introduction.html", "conclusion.html"]:
        ps = re.findall(r"(?s)<p>(.*?)</p>", re.sub(r"(?s)<details class=tech>.*?</details>", "", pages[fname]))
        counts = [len(re.findall(r"[.!?](?:\s|$)", html.unescape(re.sub(r"<[^>]+>", "", p)))) for p in ps]
        print(fname, "sentence counts per <p>:", counts)


if __name__ == "__main__":
    main()

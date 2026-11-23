"""Creep corpus: documents rendered from the creep rupture dataset plus method notes,
and template questions with known gold documents and answers.

Documents (the pattern of RAG over engineering records: data sheets, test reports, notes)
  * alloy data sheet, one per alloy: composition, number of tests, tested temperature and
    stress ranges, rupture-life range, CALPHAD descriptor ranges
  * test record, one per creep test: alloy, temperature (K and deg C), stress, time to
    rupture, Larson-Miller parameter, CALPHAD descriptors at the test temperature
  * method notes: paragraphs from the two creep-study READMEs (prose, not data)

Questions are generated from the data, so every one has a gold document set and a gold
answer. Alloys are split, not questions:
  * 6 alloys are held out of the index; questions about them must be declined
  * the other 32 alloys are split by alloy into a calibration half and a test half
Aggregation questions ("which alloy lasted longest...") are kept separately: answering
them needs a scan over many records, which retrieval of 5 passages cannot do.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "creep_tests.csv"
NOTES = ROOT / "data" / "notes"

ELEMENTS = {"ni": "nickel", "cr": "chromium", "co": "cobalt", "mo": "molybdenum", "fe": "iron",
            "al": "aluminium", "ti": "titanium", "hf": "hafnium", "nb": "niobium", "ta": "tantalum",
            "w": "tungsten", "v": "vanadium", "zr": "zirconium", "cu": "copper", "mn": "manganese",
            "c": "carbon"}
RENAME = {"System Name": "alloy", "Stress (Mpa)": "stress", "Temperature": "T",
          "Lattice Misfit": "misfit", "Amount of Gamma Prime": "gp",
          "ln(Effective Diffusivity of Gamma Phase)": "lnD", "ln(Time to rupture(H))": "lnt", "LMP_K": "lmp"}


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str
    kind: str          # sheet | record | note
    alloy: str | None
    meta: tuple = ()   # (("mpa", 552), ("T", 1033.15)) for records; used by the metadata filter


@dataclass
class Question:
    qid: str
    text: str
    qtype: str     # test_lookup | approx_lookup | composition | temp_range | descriptor | method | aggregation
    gold_docs: frozenset[str]       # empty for out-of-scope and aggregation questions
    answers: tuple[float | str, ...]
    alloy: str | None
    split: str                      # calib | test
    in_scope: bool


@dataclass
class CreepCorpus:
    docs: list[Doc]                 # indexed documents only
    questions: list[Question] = field(default_factory=list)
    held_out_alloys: tuple[str, ...] = ()

    def select(self, split: str, in_scope: bool, qtypes: set[str] | None = None) -> list[Question]:
        return [q for q in self.questions if q.split == split and q.in_scope == in_scope
                and (qtypes is None or q.qtype in qtypes)]


def load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={**RENAME, **{e.upper(): e for e in ELEMENTS}})
    df[list(ELEMENTS)] = df[list(ELEMENTS)].fillna(0.0)
    df["alloy"] = df["alloy"].str.strip()
    df["hours"] = np.exp(df["lnt"])
    df["mpa"] = df["stress"].round().astype(int)          # stresses were digitised; report whole MPa
    df["degc"] = (df["T"] - 273.15).round().astype(int)
    return df


def _hours(h: float) -> str:
    return f"{h:,.1f}" if h < 1e4 else f"{h:,.0f}"


def render_record(i: int, r) -> Doc:
    text = (f"{r.alloy} creep rupture test {i:04d}: tested at {r.T:.2f} K ({r.degc} °C) under {r.mpa} MPa "
            f"and ruptured after {_hours(r.hours)} hours (Larson-Miller parameter {r.lmp:.2f}); CALPHAD at the test "
            f"temperature gives gamma-prime fraction {r.gp:.3f}, lattice misfit {r.misfit:.4f} and ln "
            f"gamma-matrix diffusivity {r.lnD:.2f}.")
    return Doc(f"test-{i:04d}", r.alloy, text, "record", r.alloy, (("mpa", int(r.mpa)), ("T", float(r.T))))


def render_sheet(alloy: str, g: pd.DataFrame) -> Doc:
    comp = g.iloc[0]
    parts = [f"{e.capitalize()} ({ELEMENTS[e]}) {100 * comp[e]:.2f}" for e in ELEMENTS if comp[e] > 0]
    text = (f"{alloy} composition in atom percent (data sheet): {', '.join(parts)}. "
            f"{alloy} was creep tested over a temperature range of {g['T'].min():.2f} K to {g['T'].max():.2f} K "
            f"({g['degc'].min()} to {g['degc'].max()} °C), at stresses from {g['mpa'].min()} to {g['mpa'].max()} MPa, "
            f"in {len(g)} rupture tests. "
            f"{alloy} rupture lives range from {_hours(g['hours'].min())} to {_hours(g['hours'].max())} hours. "
            f"{alloy} CALPHAD gamma-prime fraction over the tested temperatures ranges from {g['gp'].min():.3f} "
            f"to {g['gp'].max():.3f}.")
    return Doc(f"sheet-{re.sub(r'[^a-z0-9]+', '-', alloy.lower()).strip('-')}", alloy, text, "sheet", alloy)


def render_notes() -> list[Doc]:
    docs = []
    for path in sorted(NOTES.glob("*.md")):
        body = "\n".join(l for l in path.read_text().splitlines()
                         if l.strip() and not l.startswith("#") and "docs.google.com" not in l)
        sents = re.split(r"(?<=[.!?])\s+", body.strip())
        for j in range(0, len(sents), 3):           # 3-sentence chunks
            docs.append(Doc(f"note-{path.stem.split('_')[0]}-{j // 3}", "",
                            " ".join(sents[j:j + 3]), "note", None))
    return docs


# hand-written questions about the method notes: (question, gold note ids, expected answer strings)
METHOD_QUESTIONS = [
    ("Which machine learning model gave the best accuracy for creep rupture life?",
     {"note-calphad-ml-creep-0", "note-creep-article-1"}, ("gradient boosting",)),
    ("How many features did the creep rupture models use?",
     {"note-calphad-ml-creep-0", "note-creep-article-0"}, ("21",)),
    ("How were the microstructural descriptors in the creep study computed?",
     {"note-calphad-ml-creep-0", "note-creep-article-0"}, ("calphad",)),
    ("By how much were features perturbed in the numerical sensitivity analysis?",
     {"note-calphad-ml-creep-1", "note-creep-article-2"}, ("10%",)),
    ("Which features had minimal impact on creep life according to SHAP?",
     {"note-creep-article-1"}, ("zr",)),
    ("What method was used to interpret the gradient boosting predictions of creep life?",
     {"note-calphad-ml-creep-1", "note-creep-article-1"}, ("shap",)),
    ("Does increasing alloying element concentration improve predicted creep life?",
     {"note-calphad-ml-creep-1", "note-creep-article-2"}, ("ti",)),
    ("What application are these Ni-based superalloy creep results aimed at?",
     {"note-calphad-ml-creep-2", "note-creep-article-3"}, ("turbine blades",)),
]
# the source notes disagree on this one: calphad-ml-creep-2 and creep-article-2 say a higher
# gamma-prime fraction improves creep resistance, creep-article-1 says it reduces creep life
CONFLICT_QUESTION = ("Does a higher gamma-prime fraction increase or reduce creep life?",
                     {"note-calphad-ml-creep-2", "note-creep-article-1", "note-creep-article-2"})


def _q_variants(rng, alloy, mpa, T, degc):
    return rng.choice([
        f"How long did {alloy} last at {mpa} MPa and {T:.2f} K?",
        f"What was the time to rupture of {alloy} under {mpa} MPa at {degc} °C?",
        f"Rupture life of {alloy} tested at {degc} °C and {mpa} MPa?",
        f"When did the {alloy} creep test at {T:.2f} K and {mpa} MPa fail?",
    ])


def build(n_holdout: int = 6, seed: int = 0, max_lookup_per_alloy: int = 40) -> CreepCorpus:
    df = load_df()
    rng = random.Random(seed)
    alloys = sorted(df.alloy.unique())
    order = alloys[:]
    rng.shuffle(order)
    held = set(order[:n_holdout])
    split = {}
    for group in ([a for a in order if a in held], [a for a in order if a not in held]):
        for k, a in enumerate(group):
            split[a] = "calib" if k % 2 == 0 else "test"

    records = {i: render_record(i, r) for i, r in enumerate(df.itertuples())}
    sheets = {a: render_sheet(a, g) for a, g in df.groupby("alloy")}
    notes = render_notes()
    docs = [d for d in records.values() if d.alloy not in held] + \
           [d for a, d in sheets.items() if a not in held] + notes
    corpus = CreepCorpus(docs, held_out_alloys=tuple(sorted(held)))
    Q = corpus.questions

    def add(qtype, text, gold, answers, alloy):
        ins = alloy not in held
        Q.append(Question(f"{qtype}-{len(Q):05d}", text, qtype, frozenset(gold if ins else ()), tuple(answers),
                          alloy, split[alloy], ins))

    for alloy, g in df.groupby("alloy"):
        # test lookups: gold = every record of this alloy at the same stress and temperature
        keys = g.groupby(["mpa", "T"]).groups
        picked = list(keys.items())
        rng.shuffle(picked)
        for (mpa, T), idx in picked[:max_lookup_per_alloy]:
            degc = int(round(T - 273.15))
            add("test_lookup", _q_variants(rng, alloy, mpa, T, degc), {records[i].doc_id for i in idx},
                [round(df.hours[i], 1) for i in idx], alloy)
        # approximate lookups: stress to the nearest 10 MPa, temperature to the nearest 10 deg C;
        # gold = every test of the alloy within 5 MPa and 5 deg C of what was asked
        for (mpa, T), idx in picked[:max_lookup_per_alloy // 2]:
            s10, c10 = int(round(mpa, -1)), int(round((T - 273.15) / 10.0) * 10)
            near = g[(abs(g.mpa - s10) <= 5) & (abs(g.degc - c10) <= 5)]
            if near.empty:
                continue
            text = rng.choice([f"Roughly how long does {alloy} last at about {s10} MPa and {c10} °C?",
                               f"Approximate rupture life of {alloy} near {s10} MPa at around {c10} °C?",
                               f"What rupture time should I expect for {alloy} at ~{s10} MPa, {c10} °C?"])
            add("approx_lookup", text, {records[i].doc_id for i in near.index},
                [round(h, 1) for h in near.hours], alloy)
        sheet = sheets[alloy].doc_id
        comp = g.iloc[0]
        for e, name in ELEMENTS.items():
            if comp[e] > 0 and e != "ni":
                text = rng.choice([f"How much {name} does {alloy} contain?",
                                   f"What is the {e.capitalize()} content of {alloy} in atom percent?"])
                add("composition", text, {sheet}, [round(100 * comp[e], 2)], alloy)
        add("temp_range", rng.choice([f"Over what temperature range was {alloy} creep tested?",
                                      f"What were the lowest and highest test temperatures for {alloy}?"]),
            {sheet}, [round(g['T'].min(), 2), round(g['T'].max(), 2)], alloy)
        for T, gt in list(g.groupby("T"))[:6]:
            add("descriptor", f"What gamma-prime fraction does CALPHAD give for {alloy} at {T:.2f} K?",
                {records[i].doc_id for i in gt.index}, [round(gt.gp.iloc[0], 3)], alloy)

    # method-note questions are about the study, not one alloy: assign alternately to calib/test
    for k, (text, gold, ans) in enumerate(METHOD_QUESTIONS):
        Q.append(Question(f"method-{k:02d}", text, "method", frozenset(gold), ans, None,
                          "calib" if k % 2 == 0 else "test", True))

    # aggregation questions: no single passage holds the answer
    hot = df[df["T"] > 1200]
    agg = [
        ("Which alloy lasted longest in tests above 1200 K?", hot.loc[hot.hours.idxmax(), "alloy"]),
        ("Which alloy has the most creep tests in the dataset?", df.alloy.value_counts().index[0]),
        ("How many creep tests were run above 800 MPa?", int((df.mpa > 800).sum())),
        ("How many distinct alloys are in the creep dataset?", df.alloy.nunique()),
        ("Which alloy has the highest gamma-prime fraction?", df.loc[df.gp.idxmax(), "alloy"]),
        ("What is the highest test temperature in the dataset?", round(df["T"].max(), 2)),
        ("How many alloys contain tungsten?", int((df.groupby("alloy").w.first() > 0).sum())),
        ("What is the median time to rupture across all tests, in hours?", round(float(df.hours.median()), 1)),
    ]
    for k, (text, ans) in enumerate(agg):
        Q.append(Question(f"aggregation-{k:02d}", text, "aggregation", frozenset(), (ans,), None, "test", True))
    return corpus


def answer_matches(answer: str | None, expected: tuple, tol: float = 0.01) -> bool:
    """True if every expected value appears in the answer (numbers within 1%)."""
    if not answer:
        return False
    a = answer.lower().replace(",", "")
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", a)]
    for e in expected:
        if isinstance(e, str):
            if e.lower() not in a:
                return False
        elif not any(math.isclose(n, float(e), rel_tol=tol, abs_tol=0.051) for n in nums):
            return False
    return True


def answer_matches_any(answer: str | None, expected: tuple) -> bool:
    """For test lookups with repeated tests: any one of the gold rupture times counts."""
    return any(answer_matches(answer, (e,)) for e in expected)

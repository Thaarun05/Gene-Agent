"""
Verification layer with two independent checks per claim:
    (1) EXISTENCE: every source_id a claim cites must exist in the provenance store, 
        a missing id is a hallucination.
    (2) SUPPORT: a SECOND, independent Claude call is shown the claim plus the ACTUAL
        content of its cited sources, and asked whether the source supports the claim (yes/partially/no).
        Anything not a clean "yes" is flagged.
"""

import json
import re
from dataclasses import dataclass

from .agent import client, MODEL
from .provenance import ProvenanceStore
from .schema import GeneDossier


@dataclass
class ClaimVerdict:
    section: str
    claim: str
    cited_ids: list[str]
    missing_ids: list[str]    # cited but not in the store
    support: str              # yes/partially/no
    reason: str=""

    @property
    def flagged(self) -> bool:
        # A real problem: a cited id is missing (hallucination), or the support
        # check returned something other than a clean "yes". Uncited sentences are
        # NOT counted here -- they are reported separately (many are harmless lead-ins).
        return bool(self.missing_ids) or self.support in ("partially", "no")



# Claim extraction
# Abbreviations whose trailing period must NOT be treated as a sentence boundary.
_ABBREVIATIONS = (
    "et al.", "e.g.", "i.e.", "etc.", "vs.", "cf.", "Fig.", "No.",
    "approx.", "Dr.", "Inc.", "Ref.",
)
# A sentence boundary = .?! then whitespace then an uppercase letter. Requiring an
# uppercase next char also avoids splitting mid-decimal (e.g. "0.998" has no space).
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.?!])\s+(?=[A-Z])")
_PROTECT = "\x00"  # stand-in for a period we must not split on


def _split_sentences(text: str) -> list[str]:
    """
    Split prose into sentences without breaking on common abbreviations
    ('et al.', 'e.g.', ...) or mid-decimal. Approximate, but far cleaner than a
    naive period-split: it stops author names like 'Tabrizi et al.' from becoming
    their own citation-less fragments.
    """
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _PROTECT))
    sentences = _SENTENCE_BOUNDARY.split(protected)
    return [s.replace(_PROTECT, ".").strip() for s in sentences if s.strip()]


def extract_claims(section) -> list[tuple[str, list[str]]]:
    """
    Split a section's content into sentence-level claims, attaching cited source_ids.
    Detect cited_ids by substring matching the section's KNOWN source_ids against each
    sentence --> avoids parsing the [source:id] text, because source_ids themselves contain
    commas, colons, and spaces.
    Sentence segmentation is approximate; that's fine, because flags are advisory
    for human review, never used to auto-edit the dossier.
    """

    claims = []
    for sentence in _split_sentences(section.content):
        cited = [source_id for source_id in section.source_ids if source_id in sentence]
        claims.append((sentence, cited))
    return claims


# Check 1: Existence (deterministic, no LLM)
def find_missing_ids(cited_ids: list[str], store: ProvenanceStore) -> list[str]:
    """
    Return cited ids NOT present in the store (likely hallucinations)
    """
    return [source_id for source_id in cited_ids if store.get(source_id) is None]



# Check 2: Support (LLM-powered reasoning)
VERIFY_SYSTEM_PROMPT = (
    "You are a strict fact-checker. You receive a CLAIM and the ACTUAL SOURCE "
    "CONTENT it cites. Decide only whether the source content supports the claim, "
    "using NO outside knowledge. 'yes' = the source fully supports the claim; "
    "'partially' = it supports part but not all, or is weaker than stated; 'no' = "
    "it does not support or contradicts the claim."
)


VERDICT_TOOL = {
    "name": "record_verdict",
    "description": "Record whether the cited source content supports the claim.",
    "input_schema": {
        "type": "object",
        "properties": {
            "support": {
                "type": "string",
                "enum": ["yes", "partially", "no"],
                "description": "yes / partially / no per the rules."
            },
            "reason": {"type": "string", "description": "One-sentence justification"}
        },
        "required": ["support", "reason"]
    }
}


def check_support(claim: str, cited_ids: list[str], store: ProvenanceStore) -> tuple[str, str]:
    """
    Ask Claude whether cited source support the claim. Returns (support, reason).
    """

    sources = {source_id: store.get(source_id) for source_id in cited_ids}
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=VERIFY_SYSTEM_PROMPT,
        tools=[VERDICT_TOOL],
        tool_choice={"type": "tool", "name": "record_verdict"},
        messages=[{
            "role": "user",
            "content": f"CLAIM:\n{claim}\n\nSOURCE CONTENT:\n{json.dumps(sources, default=str)}",
        }]
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "record_verdict":
            return block.input["support"], block.input.get("reason", "")
    
    return "unchecked", "no verdict returned"



# Orchestration + report
def verify_dossier(dossier: GeneDossier, store: ProvenanceStore) -> list[ClaimVerdict]:
    verdicts = []
    for section_name, section in dossier:
        for claim, cited in extract_claims(section):
            missing = find_missing_ids(cited, store)
            if not cited:
                support, reason = "uncited", "no [source:id] in this sentence"
            elif missing:
                support, reason = "unchecked", "cited id(s) not in the store"
            else:
                support, reason = check_support(claim, cited, store)

            verdicts.append(ClaimVerdict(section_name, claim, cited, missing, support, reason))

    return verdicts


def print_report(verdicts: list[ClaimVerdict]) -> None:
    flagged = [v for v in verdicts if v.flagged]
    uncited = [v for v in verdicts if not v.cited_ids and not v.flagged]
    passed = len(verdicts) - len(flagged) - len(uncited)

    print(f"\n{'='*60}\nVERIFICATION REPORT\n{'='*60}")
    print(
        f"Total claims: {len(verdicts)}  |  passed: {passed}  |  "
        f"FLAGGED: {len(flagged)}  |  uncited: {len(uncited)}"
    )

    if flagged:
        print("\n--- FLAGGED (needs manual review) ---")
        for v in flagged:
            issue = f"missing={v.missing_ids}" if v.missing_ids else f"support={v.support}"
            print(f"[{v.section}] {issue}")
            print(f"   claim : {v.claim}")
            print(f"   reason: {v.reason}")

    if uncited:
        print("\n--- UNCITED SENTENCES (no [source:id]; often lead-ins, review manually) ---")
        for v in uncited:
            print(f"[{v.section}] {v.claim}")

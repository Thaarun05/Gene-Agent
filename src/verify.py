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
        # Flagged if it cites something missing or the support check was not a clean yes
        return bool(self.missing_ids or self.support != "yes")



# Claim extraction
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")


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
    for sentence in _SENTENCE_SPLIT.split(section.content):
        sentence = sentence.strip()
        if not sentence:
            continue
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
                support, reason = "unchecked", "no citation in this sentence"
            elif missing:
                support, reason = "unchecked", "cited id(s) not in the store"
            else:
                support, reason = check_support(claim, cited, store)

            verdicts.append(ClaimVerdict(section_name, claim, cited, missing, support, reason))

    return verdicts


def print_report(verdicts: list[ClaimVerdict]) -> None:
    flagged = [v for v in verdicts if v.flagged]
    print(f"\n{'='*60}\nVERIFICATION REPORT\n{'='*60}")
    print(f"Total claims: {len(verdicts)}  |  Flagged: {len(flagged)}")
    for v in verdicts:
        mark = "FLAG" if v.flagged else "ok  "
        print(f"[{mark}] ({v.section}) support={v.support} missing={v.missing_ids}")
        if v.flagged:
            print(f"       claim : {v.claim}")
            print(f"       reason: {v.reason}")

"""
Synthesis layer: Reads every fact from ProvenanceStore and asks Claude to write
a six section dossier. The output is forced through the GeneDossier schema so it
is structured by construction, parsed via tool call.
"""

import json
from .agent import client, MODEL
from .provenance import ProvenanceStore
from .schema import GeneDossier


SYNTHESIS_SYSTEM_PROMPT = (
    "You are a cautious biomedical analyst writing a Huntington's-disease gene "
    "dossier STRICTLY from the evidence provided. Follow these rules exactly:\n"
    "1. Every sentence must end with its supporting source id(s) in [source:id] "
    "format. Never write a claim you cannot attribute to a provided source_id.\n"
    "2. Distinguish CAUSAL gene language (the gene causes the disease) from "
    "MODIFIER language (the gene modifies age-of-onset or severity). Do not "
    "upgrade a modifier to a cause.\n"
    "3. Flag GWAS-association-only evidence as such, and keep it separate from "
    "functional/mechanistic evidence. Do not present a statistical association as "
    "a proven mechanism.\n"
    "4. For STRING interactors, note when a high combined score is driven mainly "
    "by text-mining (tscore) or database (dscore) rather than experimental "
    "(escore) evidence — treat the former as weaker.\n"
    "5. Prefer UniProt comments that have populated `evidences` over unsupported "
    "ones when both exist for the same claim.\n"
    "Do not invent source ids. If the evidence for a section is thin, say so."
)


def _dump_evidence(store: ProvenanceStore) -> str:
    """
    Serialize every stored fact as '[source_id] <contents>' so Claude cites exact ids.
    """
    blocks = []
    for source_id in store.all_ids():
        content = store.get(source_id)
        blocks.append(f"{source_id}\n{json.dumps(content, default=str)}")
    return "\n\n".join(blocks)


EMIT_DOSSIER_TOOL = {
    "name": "emit_dossier",
    "description": "Emit a complete six-section dossier from the evidence provided.",
    "input_schema": GeneDossier.model_json_schema()
}


def synthesize_dossier(store: ProvenanceStore, gene_symbol: str) -> GeneDossier:
    user_message = (
        f"Gene: {gene_symbol}\n\n"
        "Below is ALL gathered evidence, each fact tagged with its source_id. "
        "Write the six-section dossier using only these facts.\n\n"
        f"{_dump_evidence(store)}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=16384,
        system=SYNTHESIS_SYSTEM_PROMPT,
        tools=[EMIT_DOSSIER_TOOL],
        tool_choice={"type":"tool", "name":"emit_dossier"},
        messages=[{"role":"user", "content": user_message}]
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "emit_dossier":
            return GeneDossier(**block.input)

    raise RuntimeError(f"Claude failed to emit a dossier. Response content: {response.content}")
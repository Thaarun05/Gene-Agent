import os
import json
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass, field

from . import tools
from .provenance import (
    ProvenanceStore,
    flatten_ncbi_gene,
    flatten_uniprot,
    flatten_list_source
)

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-6"

MAX_TOOL_CALLS = 15


# Tool schemas
    # name: Exactly matches function name in tools.py
    # description: Guide for Claude when to call the tool
    # input_schema:  JSON schema for args Claude must supply

GENE_SYMBOL_INPUT = {
    "type": "object",
    "properties": {
        "gene_symbol": {
            "type": "string",
            "description": "Gene symbol to query, e.g. 'HTT'"
        }
    },
    "required": ["gene_symbol"]
}


TOOLS = [
    {
        "name": "fetch_ncbi_gene",
        "description": (
            "Fetch core gene identity from NCBI Gene: official symbol, full name, "
            "aliases, chromosome location, and RefSeq summary. Always call this first — "
            "gene symbols can collide with aliases of unrelated genes (e.g. 'HTT' also "
            "matches SLC6A4), and this function performs exact-symbol disambiguation "
            "before any other source is queried."
        ),
        "input_schema": GENE_SYMBOL_INPUT,
    },
    {
        "name": "fetch_pubmed_hd_literature",
        "description": (
            "Fetch up to 10 PubMed abstracts specifically linking the gene to "
            "Huntington's disease, relevance-sorted and HD MeSH-filtered. Results are "
            "ordered by relevance, so earlier abstracts are more central to the gene's "
            "HD biology than later ones. Do not attribute a finding to this gene unless "
            "the abstract is specifically about it — an abstract may mention other genes "
            "in passing."
        ),
        "input_schema": GENE_SYMBOL_INPUT,
    },
    {
        "name": "fetch_uniprot",
        "description": (
            "Fetch reviewed UniProt protein data: function comments, subcellular "
            "localization, the Huntington-disease comment, and structural features "
            "(domains and HEAT repeats). MUST BE CALLED before "
            "fetch_proteins_variation, since it resolves the protein accession that "
            "the variation data depends on."
        ),
        "input_schema": GENE_SYMBOL_INPUT,
    },
    {
        "name": "fetch_proteins_variation",
        "description": (
            "Fetch Huntington-disease-associated protein variants from the EBI "
            "Proteins API. Requires fetch_uniprot to have been called first — the "
            "UniProt accession from that result is passed automatically by the "
            "orchestration layer. Only call this after fetch_uniprot has succeeded."
        ),
        "input_schema": GENE_SYMBOL_INPUT,
    },
    {
        "name": "fetch_string_interactors",
        "description": (
            "Fetch the top protein-protein interactors from the STRING database, "
            "each with seven evidence-channel scores: combined, experimental (escore), "
            "database (dscore), text-mining (tscore), coexpression (ascore), "
            "neighborhood (nscore), and phylogenetic (pscore). A high combined score "
            "driven mainly by tscore/dscore is weaker evidence than one driven by "
            "escore — note this distinction when interpreting results."
        ),
        "input_schema": GENE_SYMBOL_INPUT,
    },
    {
        "name": "fetch_gtex_expression",
        "description": (
            "Fetch median gene expression (TPM) across HD-relevant brain regions "
            "(caudate, putamen, nucleus accumbens, cortex) from GTEx, computed locally "
            "from sample-level data. Note that high expression in a region does not "
            "imply HD pathology localizes there — HTT itself is more highly expressed "
            "in cortex than striatum despite striatal neurodegeneration being the "
            "hallmark of HD."
        ),
        "input_schema": GENE_SYMBOL_INPUT,
    },
    {
        "name": "finish_dossier",
        "description": (
            "Signal that enough data has been gathered to build the dossier and "
            "stop the tool-call loop. A complete dossier requires data from all "
            "six sources: gene identity (NCBI), HD literature (PubMed), protein "
            "function and structure (UniProt), variant evidence (EBI Variation), "
            "protein interactions (STRING), and expression pattern (GTEx). Only "
            "call this once all six have returned successful results, or you have "
            "a specific documented reason why a source is unavailable for this gene."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Summarize what was collected from each of the six sources "
                        "and confirm all six were successfully queried, or explain "
                        "why any source was skipped."
                    )
                }
            },
            "required": ["reasoning"]
        }
    }
]


# Function registry: Session state + the name

@dataclass
class ToolCallRecord:
    call_no: int
    tool_name: str
    tool_input: dict
    source_ids: list[str]
    success: bool


@dataclass
class AgentSession:
    store: ProvenanceStore
    uniprot_accession: str | None = None # cached after UniProt runs. Injected into variation
    call_count: int = 0
    records: list[ToolCallRecord] = field(default_factory=list)
    final_reasoning: str = ""          # captured from finish_dossier
    stop_reason: str = ""              # "finished" | "end_turn" | "cap_reached"


"""
Map each tool NAME (must match schemas + tools.py) to a pair:
(the python function to call, the flatten function for its result shape)
finish_dossier is not here because it's a control signal not a data tool
"""
TOOL_REGISTRY = {
    "fetch_ncbi_gene": (tools.fetch_ncbi_gene, flatten_ncbi_gene),
    "fetch_pubmed_hd_literature": (tools.fetch_pubmed_hd_literature, flatten_list_source),
    "fetch_uniprot": (tools.fetch_uniprot, flatten_uniprot),
    "fetch_proteins_variation": (tools.fetch_proteins_variation, flatten_list_source),
    "fetch_string_interactors": (tools.fetch_string_interactors, flatten_list_source),
    "fetch_gtex_expression": (tools.fetch_gtex_expression, flatten_list_source)
}



# Guardrail Layer
def dispatch_tool(session: AgentSession, tool_name: str, tool_input: dict) -> dict:
    """
    Execute ONE data tool: run it, flatten its result, load it into the store, 
    log the call, and return a compact source_id-tagged summary to feed back 
    Claude (Claude reasons from the summary). Read full data from store later.

    Note 1: fetch_proteins_variation needs the UniProt accession. Inject it 
    from session.uniprot_accession, NEVER from Claude. If UniProt hasn't run 
    yet, we return an error telling Claude to call it first.
    """

    gene_symbol = tool_input["gene_symbol"]
    func, flatten = TOOL_REGISTRY[tool_name]

    # Special case accession dependency
    if tool_name == "fetch_proteins_variation":
        if session.uniprot_accession is None:
            return {
                "success": False,
                "error": "fetch_uniprot must be called before fetch_proteins_variation."
            }
        raw = func(gene_symbol, session.uniprot_accession)
    else:
        raw = func(gene_symbol)


    facts = flatten(raw)
    session.store.add(facts)

    if tool_name == "fetch_uniprot" and isinstance(raw, dict) and raw.get("success"):
        session.uniprot_accession = raw.get("accession")


    
    # log and build the feedback message
    source_ids = [f["source_id"] for f in facts]
    session.records.append(
        ToolCallRecord(
            call_no=session.call_count,
            tool_name=tool_name,
            tool_input=tool_input,
            source_ids=source_ids,
            success=bool(facts)
        )
    )
    return {"success": bool(facts), "count": len(source_ids), "source_ids": source_ids}


# AGENT LOOP

SYSTEM_PROMPT = (
    "You are a biomedical data-gathering agent building a Huntington's-disease "
    "gene dossier. Use the provided tools to gather evidence about the given gene "
    "from every relevant source (identity, literature, protein function, variants, "
    "interactors, expression). Call fetch_uniprot before fetch_proteins_variation. "
    "When you have gathered enough to build a complete dossier, call finish_dossier "
    "with your reasoning. Do not fabricate data; only use tool results."
)


def run_agent_loop(gene_symbol: str, store: ProvenanceStore, max_tool_calls: int = MAX_TOOL_CALLS) -> AgentSession:
    session = AgentSession(store=store)
    session.stop_reason = "cap_reached"

    messages = [
        {"role": "user", "content": f"Build a gene dossier for the gene {gene_symbol}."}
    ]

    while session.call_count < max_tool_calls:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice={"type": "auto"},
            messages=messages
        )

    
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            session.stop_reason = "end_turn"
            break

        tool_results = []
        finished = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            
            if block.name == "finish_dossier":
                finished = True
                session.final_reasoning = block.input.get("reasoning", "")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Acknowledged."
                })
                continue

            session.call_count += 1
            result = dispatch_tool(session, block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result)
            })

        
        messages.append({"role": "user", "content": tool_results})

        if finished:
            session.stop_reason = "finished"
            break

    return session

    

def print_trail(session: AgentSession) -> None:
    print(f"\n{'='*60}\nAGENT RUN TRAIL\n{'='*60}")
    print(f"Stop reason : {session.stop_reason}")
    print(f"Tool calls  : {session.call_count}")
    print(f"Facts stored: {len(session.store)}")
    for rec in session.records:
        status = "ok " if rec.success else "ERR"
        print(f"  [{rec.call_no:>2}] {status} {rec.tool_name} -> {len(rec.source_ids)} facts")
    if session.final_reasoning:
        print(f"\nfinish_dossier reasoning:\n  {session.final_reasoning}")
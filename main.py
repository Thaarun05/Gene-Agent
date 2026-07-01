from src import tools
from src.provenance import (
    ProvenanceStore,
    flatten_ncbi_gene,
    flatten_uniprot,
    flatten_list_source,
)
from src.agent import run_agent_loop, print_trail


def build_dossier_data(gene_symbol: str = "HTT") -> ProvenanceStore | None:
    """
    Runs the full Phase 1 tool pipeline for one gene and loads every
    fact into a fresh ProvenanceStore. This is the fixed-order version.
    Phase 3's agent loop will eventually replace this with dynamic
    tool selection, but this remains useful as a manual test.
    """
    print(f"\n{'='*60}")
    print(f"BUILDING DOSSIER DATA FOR: {gene_symbol}")
    print(f"{'='*60}\n")

    ncbi_gene_result = tools.fetch_ncbi_gene(gene_symbol)
    if not ncbi_gene_result.get("success"):
        print(f"FATAL: could not resolve {gene_symbol} via NCBI Gene.")
        return None

    pubmed_results = tools.fetch_pubmed_hd_literature(gene_symbol)
    uniprot_result = tools.fetch_uniprot(gene_symbol)

    variation_results = []
    if uniprot_result.get("success"):
        accession = uniprot_result.get("accession")
        variation_results = tools.fetch_proteins_variation(gene_symbol, accession)

    string_results = tools.fetch_string_interactors(gene_symbol)
    gtex_results = tools.fetch_gtex_expression(gene_symbol)

    store = ProvenanceStore()
    store.add(flatten_ncbi_gene(ncbi_gene_result))
    store.add(flatten_list_source(pubmed_results))
    store.add(flatten_uniprot(uniprot_result))
    store.add(flatten_list_source(variation_results))
    store.add(flatten_list_source(string_results))
    store.add(flatten_list_source(gtex_results))

    print(f"Total facts stored: {len(store)}\n")
    return store


def build_dossier_data_agentic(gene_symbol: str = "HTT") -> ProvenanceStore:
    """
    Phase 3 agentic version: lets Claude decide which tools to call (and in what
    order) via the tool-use loop, loading every fact into a fresh ProvenanceStore.
    Prints the reasoning trail so the agent's decisions can be audited afterward.

    Compare its fact count against build_dossier_data() (the deterministic golden
    reference) — for HTT both should land around 47 facts.
    """
    store = ProvenanceStore()
    session = run_agent_loop(gene_symbol, store)
    print_trail(session)
    return store


if __name__ == "__main__":
    store = build_dossier_data_agentic("HTT")

    if store:
        print("\nSpot check — HD disease comment:")
        import pprint
        pprint.pprint(store.get("uniprot:P42858:disease:HD"))
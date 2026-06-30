from . import tools

class ProvenanceStore:
    """
    A dict-backed provenance store. Every fact ingested is indexed by its
    unique source_id, so later layers (synthesis, verification) can trace
    any claim back to the exact data it came from.
    """

    def __init__(self):
        self._store = {}

    def add(self, facts: list[dict]):
        """
        Ingests a list of {source_id, content} facts, which are already
        flattened by the appropriate flatten_* function for that source.
        """

        for fact in facts:
            self._store[fact["source_id"]] = fact["content"]

    def get(self, source_id: str):
        """Retrieve original content for a given source_id"""
        return self._store.get(source_id)
    
    def all_ids(self) -> list[str]:
        """List every source_id currently stored --> used by Phase 5 verification"""
        return list(self._store.keys())
    
    def __len__(self):
        return len(self._store)
    


# Flatten functions --> one per tool, each knows its own result shape


def flatten_ncbi_gene(result: dict) -> list[dict]:
    """Flattens fetch_ncbi_gene's output - single flat dict, one fact"""
    if not result.get("success"):
        return []
    return [{"source_id": result["source_id"], "content": result}]


def flatten_uniprot(result: dict) -> list[dict]:
    """
    Flattens fetch_uniprot's output. Has a top-level fact, plus nested lists
    of function comments, subcellular comments, one disease comment, and
    structural features, each with its own source_id.
    """

    if not result.get("success"):
        return []

    facts = []
    facts.append({"source_id": result["source_id"], "content": result})

    for func_comment in result.get("function_comments", []):
        facts.append({"source_id": func_comment["source_id"], "content": func_comment})

    for subcell_comment in result.get("subcellular_comments", []):
        facts.append({"source_id": subcell_comment["source_id"], "content": subcell_comment})

    if result.get("hd_disease_comment"):
        hd = result["hd_disease_comment"]
        facts.append({"source_id": hd["source_id"], "content": hd})

    for struct_feat in result.get("structural_features", []):
        facts.append({"source_id": struct_feat["source_id"], "content": struct_feat})

    
    return facts


def flatten_list_source(results: list[dict]) -> list[dict]:
    """
    Shared flattener for the four sources that already return a flat
    LIST of dicts, each carrying its own source_id: PubMed, Variation, 
    STRING, and GTEx.
    """

    facts = []
    for item in results:
        if item.get("success") and "source_id" in item:
            facts.append({"source_id": item["source_id"], "content": item})

    return facts




if __name__ == "__main__":
    import pprint

    gene_symbol = "HTT"

    ncbi_gene_result = tools.fetch_ncbi_gene(gene_symbol)
    pubmed_results = tools.fetch_pubmed_hd_literature(gene_symbol)
    uniprot_result = tools.fetch_uniprot(gene_symbol)
    
    variation_results = []
    if uniprot_result.get("success"):
        accession = uniprot_result.get("accession")
        variation_results = tools.fetch_proteins_variation(gene_symbol, accession)

    string_results = tools.fetch_string_interactors(gene_symbol)
    gtex_results = tools.fetch_gtex_expression(gene_symbol)


    # Build Provenance store
    store = ProvenanceStore()

    store.add(flatten_ncbi_gene(ncbi_gene_result))
    store.add(flatten_list_source(pubmed_results))
    store.add(flatten_uniprot(uniprot_result))
    store.add(flatten_list_source(variation_results))
    store.add(flatten_list_source(string_results))
    store.add(flatten_list_source(gtex_results))


    print(f"Total facts stored: {len(store)}")
    print("\nFirst 10 source_ids:")
    for sid in store.all_ids()[:10]:
        print(f"  {sid}")

    # Spot-check: can we retrieve a specific fact correctly?
    print("\n--- Spot check: retrieving the HD disease comment ---")
    pprint.pprint(store.get("uniprot:P42858:disease:HD"))

    print("\n--- Spot check: retrieving one PubMed abstract ---")
    pprint.pprint(store.get("pubmed:26938440"))

import requests
import os
import time
import statistics
import pprint

# def test_ncbi_connection():
#     url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
#     params = {
#         "db": "gene",
#         "term": "HTT[gene] AND human[orgn]",
#         "api_key": os.getenv("NCBI_API_KEY"),
#         "retmode": "json"
        
#     }

#     response = requests.get(url, params=params)

#     data = response.json()

#     print(data)


def fetch_ncbi_gene(gene_symbol:str, organism: str = "human") -> dict: 
    """
    Fetches gene identity data from NCBI Gene for a given gene symbol.
    Returns: gene ID< official name, aliases, chromosome location, and summary text.

    Note: ESearch returns multiple IDs when symbol appears as an alias of an unrelated
    gene (ex. "HTT" also matches SLC6A4). Disambiguation is done by checking nomenclatureSymbol 
    in the Esummary response, not by using idList[0]

    API docs: https://www.ncbi.nlm.nih.gov/books/NBK25497/
    """
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    api_key = os.getenv("NCBI_API_KEY")

    params = {
        "db": "gene",
        "term": f"{gene_symbol}[gene] AND {organism}[orgn]",
        "sort": "relevance",
        "retmode": "json"
    }

    if api_key:
        params["api_key"] = api_key

    try:
        search_response = requests.get(BASE_URL + "esearch.fcgi", params=params, timeout=10) # esearch.fcgi means "Entrez Search" and is used to search the NCBI databases for records that match a given query. In this case, it is searching the "gene" database for records that match the specified gene symbol and organism.
        search_data = search_response.json()
    except Exception as e:
        return {"source": "ncbi_gene", "success": False, "error": f"Error during ESearch request: {str(e)}"}
    
    candidate_ids = search_data.get("esearchresult", {}).get("idlist", []) 

    if not candidate_ids:
        return {"source": "ncbi_gene", "success": False, "error": f"No NCBI Gene results found for {gene_symbol}"}
    

    summary_params = {
        "db": "gene",
        "id": ",".join(candidate_ids),
        "retmode": "json"
    }

    if api_key:
        summary_params["api_key"] = api_key

    try:
        summary_response = requests.get(BASE_URL + "esummary.fcgi", params=summary_params, timeout=10)
        summary_data = summary_response.json()
    except Exception as e:
        return {"source": "ncbi_gene", "success": False, "error": f"Error during ESummary request: {str(e)}"}
    

    matches = []
    # Disambiguate by checking nomenclatureSymbol in the Esummary response
    for gene_id in candidate_ids:
        entry = summary_data["result"][gene_id]
        # print(gene_id, list(entry.keys()))
        official_symbol = entry.get("nomenclaturesymbol", "")

        if official_symbol.upper() == gene_symbol.upper():
            matches.append(entry)

    
    if len(matches) != 1:
        return {
            "source": "ncbi_gene",
            "success": False,
            "error": f"Expected exactly 1 exact symbol match for {gene_symbol}, "
                     f"got {len(matches)}. Candidate IDs: {candidate_ids}"
        }


    gene_entry = matches[0]

    gene_id_final = gene_entry["uid"]


    return {
        "source": "ncbi_gene",
        "success": True,
        "gene_id": gene_id_final,
        "symbol": gene_entry.get("name"),
        "full_name": gene_entry.get("description"),
        "aliases": gene_entry.get("otheraliases"),   # returned as a comma-separated string
        "chromosome": gene_entry.get("chromosome"),
        "summary": gene_entry.get("summary"),
        "source_id": f"ncbi_gene:{gene_id_final}",   # provenance tag

    }



def fetch_pubmed_hd_literature(gene_symbol: str, max_results: int = 10) -> list[dict]:
    """
    Fetches PubMed abstracts for papers linking a gene to Huntington's disease.
    Returns: list of {pmid, source_id, abstract_text}
    
    Note 1: plain {gene_symbol}[gene] AND Huntington disease returns ~4200 results,
    beccause many papers mention the gene in passing. To get more relevant results,
    use [Title/Abstract] to restrict to papers where gene is substantially discussed,
    and use "Huntington Disease"[MeSH Terms] for controlled-vocabulary HD matching.
    sort=relevance surfaces central papers first instead of just recent ones.

    Note 2: retrieved abstract may mention a different gene in passing.
    The synthesis layer must not attribute a finding to gene_symbol unless
    it is actually about that gene specifically.

    API docs: https://www.ncbi.nlm.nih.gov/books/NBK25497/
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    api_key = os.getenv("NCBI_API_KEY")


    search_params = {
        "db": "pubmed",
        "term": f"{gene_symbol}[tiab] AND Huntington disease[MeSH Terms]",
        "sort": "relevance",
        "retmax": max_results,
        "retmode": "json"
    }

    if api_key:
        search_params["api_key"] = api_key
    

    try:
        search_response = requests.get(BASE_URL + "esearch.fcgi", params=search_params, timeout=10)
        search_data = search_response.json()
    except Exception as e:
        return [{"source": "pubmed", "success": False, "error": str(e)}]
    

    pmids = search_data.get("esearchresult", {}).get("idlist", [])

    if not pmids:
        return [{"source": "pubmed", "success": False,
                 "error": f"No PubMed results for {gene_symbol} + Huntington Disease"}]
    

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "text"
    }

    if api_key:
        fetch_params["api_key"] = api_key

    try:
        fetch_response = requests.get(BASE_URL + "efetch.fcgi", params=fetch_params, timeout=10)
        raw_text = fetch_response.text
    except Exception as e:
        return [{"source": "pubmed", "success": False, "error": str(e)}]
    

    blocks = [b.strip() for b in raw_text.split("\n\n\n") if b.strip()] # Each abstract block starts with its PMID on the first line.
    # A reliable way to split them: the blocks are separated by "\n\n\n"
    # (two blank lines). Split on that, drop any empty chunks.

    results = []
    for pmid, block in zip(pmids, blocks):
        results.append({
            "source": "pubmed",
            "success": True,
            "pmid": pmid,
            "abstract_text": block,
            "source_id": f"pubmed:{pmid}",
        }
        )


    return results



def fetch_uniprot(gene_symbol: str, organism_id: str = "9606") -> dict:
    """
    Fetches protein function, localization, domains, and disease info from UniProt.
    Returns the UniProt accession --> fetch_variation() needs it downstream.

    Note 1: Comments (FUNCTION, DISEASE, SUBCELLULAR LOCATION) live in a flat comments array, 
    each tagged with commentType. A protein can have multiple entries of same type scoped to
    different molecule values. Capture molecule alongside every comment, don't get first one.

    Note 2: Evidence quality varies per comment. Some have an evidences array with evidenceCode
    (ECO:0000269 = experimental). Others have none at all. Unsupported claim is weaker than an 
    evidence-backed one.

    Note 3: A gene can map to multiple DISEASE comments (ex. HTT maps to both Huntingotn disease AND 
    Lopes-Maciel Rodan Syndrome). Filter explicitly on diseaseID == "Huntington disease".

    Note 4: ft_domain can legitimately return empty for a real biological reason.
    HTT has zero Domain features but five Repeat features (HEAT repeats). Always
    request both ft_domain and ft_repeat, treat empty as a valid answer not an error.

    API docs: https://www.uniprot.org/api-documentation/uniprotkb
    """

    BASE_URL = "https://rest.uniprot.org/uniprotkb/search"

    params = {
        "query": f"gene:{gene_symbol} AND organism_id:{organism_id} AND reviewed:true",
        "fields": "accession,gene_names,protein_name,cc_function,cc_subcellular_location,cc_disease,ft_domain,ft_repeat",
        "format": "json",
        "size": 1
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()
    except Exception as e:
        return {"source": "uniprot", "success": False, "error": str(e)}


    results = data.get("results", [])

    if not results:
        return {"source": "uniprot", "success": False,
                "error": f"No reviewed UniProt entry found for {gene_symbol}"}
    

    entry = results[0]
    accession = entry.get("primaryAccession")

    function_comments = []
    subcellular_comments = []
    hd_disease_comment = None
    all_disease_names = []


    for comment in entry.get("comments", []):
        comment_type = comment.get("commentType")
        molecule = comment.get("molecule", "default")

        if comment_type == "FUNCTION":
            texts = comment.get("texts", [])
            for text_obj in texts:
                function_comments.append({
                    "molecule": molecule,
                    "text": text_obj.get("value"),
                    "evidences": text_obj.get("evidences", []),
                    "source_id": f"uniprot:{accession}:function:{molecule}"
                })

        elif comment_type == "SUBCELLULAR LOCATION":
            locations = comment.get("subcellularLocations", [])
            for loc in locations:
                subcellular_comments.append({
                    "molecule": molecule,
                    "location": loc.get("location", {}).get("value"),
                    "evidences": loc.get("location", {}).get("evidences", []),
                    "source_id": f"uniprot:{accession}:subcellular:{molecule}"
                })

        elif comment_type == "DISEASE":
            disease = comment.get("disease", {})
            disease_name = disease.get("diseaseId", "")
            all_disease_names.append(disease_name)

            # Keep only HD-specific one
            if disease_name == "Huntington disease":
                hd_disease_comment = {
                    "disease_id": disease_name,
                    "description": disease.get("description"),
                    "mim_id": disease.get("diseaseCrossReference", {}).get("id"),
                    "evidences": disease.get("evidences", []),
                    "source_id": f"uniprot:{accession}:disease:HD" 
                }


    # Parse features array (domains and repeats)
    # Want items where type == "Domain" OR type == "Repeat"
    structural_features = []
    for feature in entry.get("features", []):
        if feature.get("type") in ("Domain", "Repeat"):
            structural_features.append({
                "type": feature.get("type"),
                "description": feature.get("description"),
                "start": feature.get("location", {}).get("start", {}).get("value"),
                "end": feature.get("location", {}).get("end", {}).get("value"),
                "source_id": f"uniprot:{accession}:feature:{feature.get('description')}"
            })

    
    return {
        "source": "uniprot",
        "success": True,
        "accession": accession,
        "function_comments": function_comments,
        "subcellular_comments": subcellular_comments,
        "hd_disease_comment": hd_disease_comment,
        "all_disease_names": all_disease_names,   # surface all of them — flag if >1
        "structural_features": structural_features,
        "source_id": f"uniprot:{accession}",
    }



def fetch_proteins_variation(gene_symbol: str, uniprot_accession: str, disease_term: str = "Huntington") -> list[dict]:
    """
    Fetches disease-associated protein variants from EBI Proteins API.
    REQUIRES a UniProt accession --> call fetch_uniprot first and pass its result["accession] here

    Note 1: Disease param does loose text matching, NOT a strict filter. Ex.) Querying disease=Huntington
    for HTT returned a variant tagged with 3 disease labels simultaneously (LOMARS, Huntington disease,
    Lopes-Maciel-Rodan syndrome). Refilter in code, don't use API param alone.

    Note 2: association[].disease is a boolean meaning "does this variant's clinical significance support
    it as disease causing". It is NOT just "is this variant linked to HD." A Likely Benign variant gets 
    disease=False even if it appears in an HD-tagged entry. Use this as the primary pathogenicity signal.

    Note 3: consequenceType distinguishes mechanism categories. 
    insertion/frameshift/inframe deletion = repeat-tract variants (core HD mechanism)
    missense = point mutations elsewhere in the protein (different evidence class)

    API docs: https://www.ebi.ac.uk/proteins/api/doc/
    """

    BASE_URL = "https://www.ebi.ac.uk/proteins/api/variation"

    params = {
        "accession": uniprot_accession,
        "disease": disease_term
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()
    except Exception as e:
        return [{"source": "variation", "success": False, "error": str(e)}]
    
    if not data:
        return [{"source": "variation", "success": False,
                 "error": f"No variation data found for accession {uniprot_accession}"}]
    

    entry = data[0]
    raw_variants = entry.get("features", [])

    results = []

    for variant in raw_variants:
        associations = variant.get("association", [])
        hd_associations = [association for association in associations if "Huntington" in association.get("name", "")]
        if not hd_associations:
            continue

        is_disease_causing = any(a.get("disease", False) for a in hd_associations)

        clinical_signals = variant.get("clinicalSignificances", [])
        clinical_signal_label = clinical_signals[0].get("type") if clinical_signals else None

        xrefs = variant.get("xrefs", [])
        clinvar_ids = [x.get("id") for x in xrefs if x.get("name") == "ClinVar"]
        dbsnp_ids = [x.get("id") for x in xrefs if x.get("name") == "dbSNP"]

        # pop_freqs = variant.get("populationFrquencies", [])

        variant_id = variant.get("ftId") or variant.get("id") or f"pos_{variant.get('begin')}"

        results.append({
            "source": "variation",
            "success": True,
            "variant_id": variant_id,
            "location": f"{variant.get('begin')}-{variant.get('end')}",
            "consequence_type": variant.get("consequenceType"),
            "wild_type": variant.get("wildType"),
            "mutated_type": variant.get("mutatedType"),
            "clinical_significance": clinical_signal_label,
            "is_disease_causing": is_disease_causing,
            "pop_freqs": variant.get("populationFrequencies", []),
            "clinvar_ids": clinvar_ids,
            "dbsnp_ids": dbsnp_ids,
            "source_id": f"variation:{uniprot_accession}:{variant_id}"
        })


    return results



def fetch_string_interactors(gene_symbol: str, species: str = "9606", limit: int = 10) -> list[dict]:
    """
    Fetches top protein-protein interactors from STRING database.
    Uses resolve-first pattern: get_string_ids -> interaction parteners.

    Note 1: Always resolve gene symbol to a STRING ID first. 
    Note 2: Always set caller_identity
    Note 3: Wait 1 second between calls. Never run STRING calls in parallel.
    Note 4: Capture all seven score channels, not just combined score. Two interactors can have identical 
    combined scores with completely different evidence profiles (one experimental, one text-mining). The 
    synthesis layer needs this distinction

    API docs: https://string-db.org/help/api/
    """

    BASE_URL = "https://string-db.org/api/json"
    CALLER_ID = "gene_agent_tool"

    resolve_params = {
        "identifiers": gene_symbol,
        "species": species,
        "caller_identity": CALLER_ID
    }


    try:
        resolve_response = requests.get(BASE_URL + "/get_string_ids", params=resolve_params, timeout=10)
        resolve_data = resolve_response.json()
    except Exception as e:
        return [{"source": "string", "success": False, "error": str(e)}]
    
            
    if not resolve_data:
        return [{"source": "string", "success": False,
                 "error": f"STRING could not resolve gene symbol: {gene_symbol}"}]
    

    string_id = resolve_data[0].get("stringId")

    time.sleep(1) # 1-second dealy between STRING calls


    interaction_partners_params = {
        "identifiers": string_id,
        "species": species,
        "limit": limit,
        "caller_identity": CALLER_ID
    }

    
    try:
        interaction_partners_response = requests.get(BASE_URL + "/interaction_partners", params=interaction_partners_params, timeout=10)
        partners_data = interaction_partners_response.json()
    except Exception as e:
        return [{"source": "string", "success": False, "error": str(e)}]
    

    # Package each interactor with all seven score channels
    results = []
    
    for partner in partners_data:
        results.append({
            "source": "string",
            "success": True,
            "partner_name": partner.get("preferredName_B"),
            "partner_string_id": partner.get("stringId_B"),
            "scores": {
                "combined": partner.get("score"),
                "experimental": partner.get("escore"),
                "database": partner.get("dscore"),
                "textmining": partner.get("tscore"),
                "coexpression": partner.get("ascore"),
                "neighborhood": partner.get("nscore"),
                "fusion": partner.get("fscore"),
                "phylogenetic": partner.get("pscore")
            },
            "source_id": f"string:{string_id}:{partner.get('stringId_B')}"
        })

    return results



def fetch_gtex_expression(gene_symbol: str, dataset_id: str = "gtex_v8", tissue_ids: list = None) -> list[dict]:
    """
    Fetches tissue_level expression data from GTEx for HD-relevant brain regions.

    Note 1: /expression/medianGeneExpression is confirmed broken for at least HTT —
    returns empty data under both gtex_v8 and gtex_v10, despite HTT having abundant
    real expression data (confirmed via GAPDH control, which worked fine). Use
    /expression/geneExpression (sample-level) instead and compute the median
    ourselves in Python from each tissue's raw per-sample values.

    Note 2: GTEx requires a versioned GENCODE ID (e.g. ENSG00000197386.9), not a
    plain gene symbol. Resolve via /reference/gene first.

    Note 3: Per the live OpenAPI spec, both geneId (resolve call) and gencodeId
    (expression call) are ARRAY-typed params, even for a single gene --> wrap in a
    list, e.g. ["HTT"] not "HTT".

    Note 4: tissueSiteDetailId accepts multiple repeated query params in ONE call,
    no need to loop per tissue. Pass tissue_ids directly as a list.

    Note 5: geneExpression's datasetId defaults to gtex_v10 if not specified -->
    we explicitly override to gtex_v8 since v10 has known issues for this gene.

    Note 6: GTEx asks for sequential, non-parallel querying. This function makes
    exactly 2 calls —-> add a 1-second delay between them, and never run multiple
    genes' GTEx calls concurrently in any future batch code.

    API docs: https://gtexportal.org/api/v2/redoc
    """

    BASE_URL = "https://gtexportal.org/api/v2/"

    if tissue_ids is None:
        tissue_ids = [
            "Brain_Caudate_basal_ganglia",
            "Brain_Putamen_basal_ganglia",
            "Brain_Nucleus_accumbens_basal_ganglia",
            "Brain_Cortex"
        ]

    resolve_params = {
        "geneId": [gene_symbol],
        "gencodeVersion": "v26"
    }

    try:
        resolve_response = requests.get(BASE_URL + "reference/gene", params=resolve_params, timeout=10)
        resolve_data = resolve_response.json()
    except Exception as e:
        return [{"source": "gtex", "success": False, "error": str(e)}]
    

    genes_found = resolve_data.get("data",[])

    if not genes_found:
        return [{"source": "gtex", "success": False,
                 "error": f"GTEx could not resolve gene symbol: {gene_symbol}"}]
    

    gencode_id = genes_found[0].get("gencodeId")

    time.sleep(1)

    expression_params = {
        "gencodeId": [gencode_id],
        "datasetId": dataset_id,
        "tissueSiteDetailId": tissue_ids,
        "itemsPerPage": 1000
    }


    try:
        expression_response = requests.get(BASE_URL + "expression/geneExpression", params=expression_params, timeout=10)
        expression_data = expression_response.json()
    except Exception as e:
        return [{"source": "gtex", "success": False, "error": str(e)}]
    
    raw_records = expression_data.get("data", [])

    results = []
    for record in raw_records:
        tissue = record.get("tissueSiteDetailId")
        sample_values = record.get("data", [])

        if not sample_values:
            continue

        median_tpm = statistics.median(sample_values)

        results.append({
            "source": "gtex",
            "success": True,
            "tissue": tissue,
            "median_tpm": median_tpm,
            "n_samples": len(sample_values),
            "source_id": f"gtex:{gencode_id}:{tissue}"
        })

    
    return results


def run_tools_test(gene_symbol: str = "HTT"):
    print(f"\n{'='*60}")
    print(f"TESTING GENE DOSSIER PIPELINE FOR: {gene_symbol}")
    print(f"{'='*60}\n")

    # ── 1. NCBI Gene ────────────────────────────────────────────────────────
    print("--- 1. NCBI Gene ---")
    ncbi_gene_result = fetch_ncbi_gene(gene_symbol)
    pprint.pprint(ncbi_gene_result)

    # Stop early if the core gene identity lookup fails — nothing downstream
    # makes sense without this succeeding.
    if not ncbi_gene_result.get("success"):
        print(f"\nFATAL: could not resolve {gene_symbol} via NCBI Gene. Stopping.")
        return

    # ── 2. NCBI PubMed ──────────────────────────────────────────────────────
    print("\n--- 2. NCBI PubMed (HD literature) ---")
    pubmed_results = fetch_pubmed_hd_literature(gene_symbol)
    print(f"Retrieved {len(pubmed_results)} abstracts")
    pprint.pprint(pubmed_results[0] if pubmed_results else "No results")

    # ── 3. UniProt ──────────────────────────────────────────────────────────
    print("\n--- 3. UniProt ---")
    uniprot_result = fetch_uniprot(gene_symbol)
    pprint.pprint(uniprot_result)

    # ── 4. EBI Proteins API — Variation ─────────────────────────────────────
    # Depends on UniProt's accession — only run if step 3 succeeded
    print("\n--- 4. EBI Proteins API (Variation) ---")
    if uniprot_result.get("success"):
        accession = uniprot_result.get("accession")
        variation_results = fetch_proteins_variation(gene_symbol, accession)
        print(f"Retrieved {len(variation_results)} HD-associated variants")
        pprint.pprint(variation_results)
    else:
        print("SKIPPED — UniProt fetch failed, no accession available")

    # ── 5. STRING ───────────────────────────────────────────────────────────
    print("\n--- 5. STRING (interactors) ---")
    string_results = fetch_string_interactors(gene_symbol)
    print(f"Retrieved {len(string_results)} interactors")
    pprint.pprint(string_results)

    # ── 6. GTEx ─────────────────────────────────────────────────────────────
    print("\n--- 6. GTEx (expression) ---")
    gtex_results = fetch_gtex_expression(gene_symbol)
    print(f"Retrieved {len(gtex_results)} tissue records")
    pprint.pprint(gtex_results)

    print(f"\n{'='*60}")
    print("PIPELINE TEST COMPLETE")
    print(f"{'='*60}\n")




if __name__ == "__main__":
    run_tools_test("HTT")

    # test_ncbi_connection()
    # ncbi_gene_result = fetch_ncbi_gene("HTT")
    
    # pprint.pprint(ncbi_gene_result)

    # pubmed_hd_lit_result = fetch_pubmed_hd_literature("HTT")
    # pprint.pprint(pubmed_hd_lit_result)

    # fetch_uniprot_result = fetch_uniprot("HTT")
    # pprint.pprint(fetch_uniprot_result)

    # fetch_proteins_variation_results = fetch_proteins_variation("HTT", "P42858")
    # pprint.pprint(fetch_proteins_variation_results)

    # fetch_string_interactors_results = fetch_string_interactors("HTT")
    # pprint.pprint(fetch_string_interactors_results)

    # fetch_gtex_expr_results = fetch_gtex_expression("HTT")
    # pprint.pprint(fetch_gtex_expr_results)
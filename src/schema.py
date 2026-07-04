"""
Structured dossier schema.
Each section carries prose (content) plus the exact source_ids backing it, so the
verification layer can check every claim against the provenance store.
These pydantic models get converted to a JSON schema and handed to Claude as a
FORCED tool so the model's output is structured by construction, not parsed
out of free text.
"""

from pydantic import BaseModel, Field

class DossierSection(BaseModel):
    content: str = Field(
        ..., 
        description=(
            "Synthesized prose for this section. Every sentence must end with "
            "its supporting source_id(s) in [source:id] format, exactly as "
            "stored in the provenance store. If a claim cannot be traced to a "
            "specific source_id, write 'UNVERIFIED:' before it instead of "
            "omitting it or citing a guess."
        )
    )
    source_ids: list[str] = Field(
        ...,
        description=(
            "Every source_id cited anywhere in content, copied exactly as they "
            "appear in the provenance store (e.g. 'uniprot:P42858:disease:HD', "
            "'pubmed:26938440', 'string:9606.ENSP00000347184:9606.ENSP00000334002'). "
            "Do not paraphrase or reconstruct these — the verification layer does "
            "an exact-match lookup against the store."
        )
    )


class GeneDossier(BaseModel):
    gene_summary: DossierSection = Field(
        ...,
        description=(
            "Official gene identity: symbol, full name, chromosome location, "
            "aliases, and the RefSeq summary. Draw from ncbi_gene source_ids. "
            "Note if the gene is causal for HD (variants directly cause disease) "
            "or a modifier (variants alter onset/severity but are not causative)."
        )
    )
    protein_function: DossierSection = Field(
        ...,
        description=(
            "Protein function, subcellular localization, and structural features "
            "(domains, repeats). Draw from uniprot source_ids. Distinguish "
            "evidence-backed function comments (ECO:0000269, experimental) from "
            "unsupported ones — the evidences array will be empty for the latter. "
            "Note molecule scoping where relevant (e.g. full-length vs. cleaved fragment)."
        )
    )
    expression_pattern: DossierSection = Field(
        ...,
        description=(
            "Tissue-level expression in HD-relevant brain regions (caudate, putamen, "
            "nucleus accumbens, cortex), reported as median TPM with sample count. "
            "Draw from gtex source_ids. Note that high expression does not imply "
            "HD pathology localizes to that region."
        )
    )
    interactions: DossierSection = Field(
        ...,
        description=(
            "Top protein-protein interactors from STRING, ranked by combined score. "
            "Draw from string source_ids. For each key interactor, note whether the "
            "high score is driven by experimental evidence (escore) or mainly by "
            "text-mining/database curation (tscore/dscore) — this distinction "
            "matters for how confidently the interaction can be stated."
        )
    )
    variant_evidence: DossierSection = Field(
        ...,
        description=(
            "Disease-associated protein variants from EBI Proteins API. Draw from "
            "variation source_ids. Separate repeat-tract variants (insertion, "
            "frameshift, inframe deletion at the polyQ region) from missense "
            "variants elsewhere in the protein — these are mechanistically distinct. "
            "For each variant, note is_disease_causing flag and clinical significance. "
            "Flag any 'Pathogenic' call on a variant with high population frequency "
            "as warranting extra scrutiny."
        )
    )
    hd_literature: DossierSection = Field(
        ...,
        description=(
            "Summary of the most relevant PubMed literature linking this gene to HD. "
            "Draw from pubmed source_ids. Earlier results are relevance-sorted and "
            "more central to the gene's HD biology. A finding is about this gene only "
            "if the paper specifically investigates it — not if another gene is "
            "mentioned in passing. Distinguish GWAS/association evidence from "
            "functional/mechanistic evidence; the latter is stronger."
        )
    )
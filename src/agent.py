import os
from dotenv import load_dotenv
import anthropic

from . import tools
from . provenance import(
    ProvenanceStore,
    flatten_ncbi_gene,
    flatten_uniprot,
    flatten_list_source
)

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-6"

MAX_TOOL_CALLS = 15




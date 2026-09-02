from typing import Dict

from .models import EntityType

# Mapping between spaCy tags and our EntityType values
SPACY_TO_ENTITY_TYPE = {
    "PERSON": EntityType.PERSON,
    "ORG": EntityType.ORGANIZATION,
    "GPE": EntityType.LOCATION,
    "LOC": EntityType.LOCATION,
    "PRODUCT": EntityType.PRODUCT,
    "EVENT": EntityType.EVENT,
    "DATE": EntityType.DATE,
    "MONEY": EntityType.FINANCIAL,
    "PERCENT": EntityType.FINANCIAL,
    "LAW": EntityType.CONCEPT,
    "FAC": EntityType.LOCATION,
    "WORK_OF_ART": EntityType.CONCEPT,
}

# Technology patterns (regex, complementary to spaCy).
# These are mostly proper nouns shared across languages, but may need
# extending for domain-specific or non-Latin-script tech terms.
# Pass extra_technology_patterns to __init__ to extend at runtime.
TECHNOLOGY_PATTERNS = [
    r'\b(Neo4j|MongoDB|PostgreSQL|MySQL|Redis|Elasticsearch)\b',
    r'\b(Kubernetes|Docker|Terraform|Ansible)\b',
    r'\b(Python|Java|Go|Rust|TypeScript|JavaScript)\b',
    r'\b(TensorFlow|PyTorch|LangChain|spaCy)\b',
    r'\b(RAG|GraphRAG|Vector Database|LLM|GPT)\b',
    r'\b(AWS|Azure|GCP|Cloud)\b',
]

# Max entity count for generating CO_OCCURS_WITH pairs.
# N entities → N*(N-1)/2 pairs: beyond this threshold, the cost is too high.
MAX_CO_OCCURRENCE_ENTITIES = 10

# spaCy long-text chunking (NER on very large documents)
# - segments larger than this are split at paragraph boundaries first, then
#   hard-split; parser/ner memory grows superlinearly with text length
#   (spaCy discussion #9170 / #11985), so keep segments well under the
#   pipeline's max_length (default 10**6).
SPACY_MAX_SEGMENT_CHARS = 100_000
# nlp.pipe batch size; worst-case in-flight chars = batch_size * segment cap.
SPACY_PIPE_BATCH_SIZE = 8

# Maps spaCy verb lemmas to semantic relation type labels.
# Unlisted verbs fall back to "RELATED_TO".
VERB_TO_RELATION_TYPE: Dict[str, str] = {
    "use": "USES",
    "utilize": "USES",
    "employ": "USES",
    "leverage": "USES",
    "create": "CREATES",
    "build": "CREATES",
    "make": "CREATES",
    "generate": "CREATES",
    "develop": "DEVELOPS",
    "write": "DEVELOPS",
    "code": "DEVELOPS",
    "implement": "IMPLEMENTS",
    "support": "SUPPORTS",
    "provide": "SUPPORTS",
    "extend": "EXTENDS",
    "inherit": "INHERITS",
    "derive": "INHERITS",
    "include": "INCLUDES",
    "contain": "CONTAINS",
    "embed": "INCLUDES",
    "integrate": "INTEGRATES",
    "connect": "INTEGRATES",
    "depend": "DEPENDS_ON",
    "require": "DEPENDS_ON",
    "need": "DEPENDS_ON",
    "define": "DEFINES",
    "represent": "REPRESENTS",
    "describe": "DEFINES",
    "replace": "REPLACES",
    "supersede": "REPLACES",
    "call": "CALLS",
    "invoke": "CALLS",
    "be": "IS_A",
    "design": "DESIGNS",
    "name": "NAMED",
    "become": "BECOMES",
    "produce": "PRODUCES",
    "cause": "CAUSES",
    "lead": "LEADS_TO",
    "affect": "AFFECTS",
    "prevent": "PREVENTS",
    "help": "HELPS",
    "improve": "IMPROVES",
    "reduce": "REDUCES",
    "increase": "INCREASES",
}

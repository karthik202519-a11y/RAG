"""Registry entries for chunking methods (stage: chunking).

Implemented = runs for real via studio.chunking.methods. Unavailable entries describe
the method for learning but cannot be executed yet, and say why.
"""
from .base import MethodSpec, ParamSpec, registry
from ..chunking.tokens import TOKENIZERS

C = "chunking"
STAGE = "Chunking"

TOKENIZER = ParamSpec(name="tokenizer", label="Tokenizer", type="select", default="cl100k_base", options=TOKENIZERS,
                      help="Used to measure chunk size in tokens. cl100k_base matches OpenAI text-embedding-3 models.")


def size_param(unit: str, default: int, lo: int, hi: int, help_: str = "") -> ParamSpec:
    return ParamSpec(name="size", label=f"Chunk size ({unit})", type="int", default=default, min=lo, max=hi,
                     unit=unit, help=help_ or f"Target maximum chunk length in {unit}.")


def overlap_param(unit: str, default: int, hi: int, name: str = "overlap") -> ParamSpec:
    return ParamSpec(name=name, label=f"Overlap ({unit})", type="int", default=default, min=0, max=hi, unit=unit,
                     help="Text repeated at the start of the next chunk so a fact near a boundary stays whole in "
                          "at least one chunk. Must be smaller than the chunk size.")


def _planned(id_: str, name: str, summary: str, how: str, adv: list[str], dis: list[str], reason: str,
             requires: list[str] | None = None, creds: bool = False) -> None:
    registry.register(MethodSpec(
        id=id_, category=C, stage=STAGE, name=name, summary=summary, how_it_works=how, advantages=adv,
        disadvantages=dis, status="unavailable", unavailable_reason=reason, requires=requires or [],
        needs_cloud_credentials=creds))


registry.register(MethodSpec(
    id="fixed_char", category=C, stage=STAGE, name="Fixed-size (characters)",
    summary="Cut the text every N characters, no matter what the text says.",
    how_it_works="Slides a window of `size` characters over the document; each window starts `size - overlap` "
                 "characters after the previous one. No awareness of words, sentences or layout.",
    advantages=["Trivial and fully predictable", "Fast; no tokenizer needed", "Useful as a baseline to beat"],
    disadvantages=["Cuts mid-word and mid-sentence", "Splits tables and lists arbitrarily",
                   "Characters are not tokens, so token limits are only approximate"],
    suitable_documents=["Uniform plain text", "Baseline comparisons"],
    use_cases=["Establishing a baseline before trying smarter methods"],
    params=[size_param("characters", 1500, 100, 20000), overlap_param("characters", 150, 5000)],
    inputs=["canonical text"], outputs=["chunks with character spans"],
    performance_notes="Fastest method; linear in document length.", cost_notes="No model calls.",
    references=["https://python.langchain.com/docs/concepts/text_splitters/"],
    limitations=["Ignores every kind of structure"], compare_with=["fixed_token", "recursive"],
    recommended_experiment="Chunk one document with this and with Recursive at a similar size; compare how many "
                           "chunks end mid-sentence in the statistics panel."))

registry.register(MethodSpec(
    id="fixed_token", category=C, stage=STAGE, name="Fixed-size (tokens)",
    summary="Cut the text every N tokens, so chunk size matches what the model actually reads.",
    how_it_works="Tokenizes the whole document with the chosen tokenizer, then takes windows of `size` tokens, "
                 "each starting `size - overlap` tokens after the previous. Offsets map each window back to "
                 "characters.",
    advantages=["Chunk size is exact in the unit embedding models bill and limit", "Predictable index size"],
    disadvantages=["Cuts mid-sentence and mid-word", "Structure-blind"],
    suitable_documents=["Uniform prose"], use_cases=["Baseline; hard token limits"],
    params=[size_param("tokens", 400, 20, 8000), overlap_param("tokens", 40, 4000), TOKENIZER],
    inputs=["canonical text"], outputs=["chunks with character spans"],
    requires=["tiktoken encodings (downloaded on first use)"],
    performance_notes="Linear; tokenizing is fast.", cost_notes="No model calls; the resulting chunks are billed "
    "when embedded.",
    limitations=["Overlap must be smaller than size"], compare_with=["fixed_char", "recursive", "sliding_window"],
    recommended_experiment="Run 200/0, 400/40 and 800/80 on your documents and compare retrieval metrics in the "
                           "Evaluation Center."))

registry.register(MethodSpec(
    id="sliding_window", category=C, stage=STAGE, name="Sliding window (token stride)",
    summary="Overlapping token windows defined by a stride instead of an overlap.",
    how_it_works="Window of `size` tokens advanced by `stride` tokens. Overlap = size - stride. Equivalent to "
                 "fixed-token chunking, but expressed as stride for experiments on window movement.",
    advantages=["Dense coverage; a fact is likely whole in some window", "Easy to reason about"],
    disadvantages=["Large duplication when the stride is small", "More vectors to embed and store"],
    use_cases=["Short factual lookups where boundary losses hurt"],
    params=[size_param("tokens", 400, 20, 8000),
            ParamSpec(name="stride", label="Stride (tokens)", type="int", default=200, min=1, max=8000, unit="tokens",
                      help="How far the window moves each step. Smaller stride = more overlap = more chunks."),
            TOKENIZER],
    inputs=["canonical text"], outputs=["chunks with character spans"],
    performance_notes="Chunk count grows as size/stride.", cost_notes="Embedding cost scales with duplication.",
    compare_with=["fixed_token"],
    recommended_experiment="Compare stride = size (no overlap) with stride = size/2 and look at duplication in the "
                           "statistics panel versus retrieval hit rate."))

registry.register(MethodSpec(
    id="recursive", category=C, stage=STAGE, name="Recursive splitting",
    summary="Try to cut at paragraph breaks first, then lines, then sentences, then words, only as needed.",
    how_it_works="Splits the text at the coarsest separator (blank line), recursing into any piece still larger than "
                 "`size` with the next separator (newline, '. ', space), then a hard cut. Small pieces are merged "
                 "back up to `size`, repeating trailing pieces as overlap.",
    advantages=["Respects the text's natural shape", "Rarely cuts mid-sentence", "Works on any text"],
    disadvantages=["Blind to headings, tables and page layout", "Pieces vary in size"],
    suitable_documents=["Articles, reports, plain text and Markdown"], use_cases=["A strong general-purpose default"],
    params=[size_param("tokens", 400, 20, 8000),
            ParamSpec(name="unit", label="Size unit", type="select", default="tokens",
                      options=["tokens", "characters"], help="What `size` and `overlap` measure."),
            overlap_param("tokens", 40, 4000), TOKENIZER],
    inputs=["canonical text"], outputs=["chunks with character spans"],
    performance_notes="Near linear.", cost_notes="No model calls.",
    limitations=["Does not know where one section ends and another begins"],
    compare_with=["fixed_token", "structure_aware"],
    recommended_experiment="Compare against Structure-aware on a document with tables; look at how table rows are cut."))

registry.register(MethodSpec(
    id="sentence", category=C, stage=STAGE, name="Sentence-based",
    summary="Group whole sentences until the chunk is full. Never cut inside a sentence.",
    how_it_works="Splits text into sentences with a punctuation rule, then packs consecutive sentences up to `size` "
                 "tokens. Overlap repeats the last N sentences. A sentence longer than `size` is cut by tokens.",
    advantages=["Chunks read naturally", "No mid-sentence cuts"],
    disadvantages=["Ignores sections", "Rule-based splitting mishandles abbreviations (e.g.) and tables"],
    suitable_documents=["Prose"], use_cases=["Q&A over narrative text"],
    params=[size_param("tokens", 400, 20, 8000),
            ParamSpec(name="overlap_sentences", label="Overlap (sentences)", type="int", default=1, min=0, max=10,
                      unit="sentences", help="Number of trailing sentences repeated in the next chunk."),
            TOKENIZER],
    inputs=["canonical text"], outputs=["chunks with character spans"],
    limitations=["No language-specific sentence model; English punctuation rules"],
    compare_with=["recursive", "paragraph"],
    recommended_experiment="Compare overlap 0 and 2 sentences on questions whose answer spans two sentences."))

registry.register(MethodSpec(
    id="paragraph", category=C, stage=STAGE, name="Paragraph-based",
    summary="Group whole paragraphs (the document's own blocks) until the chunk is full.",
    how_it_works="Treats each parsed block as one paragraph, packs consecutive paragraphs up to `size` tokens, and "
                 "repeats the last N paragraphs as overlap. An oversized paragraph falls back to sentence splitting.",
    advantages=["Keeps an author's paragraph intact", "Overlap is meaningful, whole-paragraph context"],
    disadvantages=["Depends on good block extraction", "Tables are treated as one paragraph"],
    suitable_documents=["Reports, handbooks, articles"], use_cases=["Long-form prose"],
    params=[size_param("tokens", 400, 20, 8000),
            ParamSpec(name="overlap_paragraphs", label="Overlap (paragraphs)", type="int", default=1, min=0, max=5,
                      unit="paragraphs", help="Trailing paragraphs repeated in the next chunk."),
            TOKENIZER],
    inputs=["parsed blocks"], outputs=["chunks with character spans"],
    compare_with=["sentence", "structure_aware"],
    recommended_experiment="Compare with Sentence-based on a handbook; check whether paragraph-level overlap helps."))

registry.register(MethodSpec(
    id="markdown_header", category=C, stage=STAGE, name="Markdown header-based",
    summary="Cut at Markdown headings (#, ##, ...), then split any long section.",
    how_it_works="Finds lines starting with 1-6 '#' characters up to `max_level`, makes each section one chunk, and "
                 "splits sections longer than `size` recursively. Optionally repeats the heading on each part.",
    advantages=["Chunks match the author's sections", "Heading context can be kept with each part"],
    disadvantages=["Needs real Markdown headings; PDFs extracted by our parser have none",
                   "Section sizes vary"],
    suitable_documents=["Markdown, README and documentation files"], use_cases=["Technical docs"],
    params=[size_param("tokens", 400, 20, 8000), overlap_param("tokens", 40, 4000),
            ParamSpec(name="max_level", label="Deepest heading level", type="int", default=3, min=1, max=6,
                      help="Headings deeper than this do not start a new chunk."),
            ParamSpec(name="include_header", label="Repeat heading on split parts", type="bool", default=True,
                      help="Prepend the section heading to every part of a split section."),
            TOKENIZER],
    inputs=["Markdown text"], outputs=["chunks with character spans"],
    limitations=["On a document with no Markdown headers it falls back to recursive splitting and says so"],
    compare_with=["recursive", "structure_aware"],
    recommended_experiment="Upload a Markdown file and compare with Recursive."))

registry.register(MethodSpec(
    id="page_aware", category=C, stage=STAGE, name="Page-aware",
    summary="One chunk per page (or per few pages), so every chunk maps to exact pages.",
    how_it_works="Uses the parser's page boundaries. `pages_per_chunk` pages become one chunk; if `max_tokens` is "
                 "set, over-long pages are re-split on sentence boundaries without crossing the group.",
    advantages=["Simple, exact page citations", "Good when the page is the natural unit (slides, forms)"],
    disadvantages=["Pages cut sections and tables in half", "Very uneven sizes"],
    suitable_documents=["Slides, forms, scanned pages"], use_cases=["Citation-heavy workflows"],
    params=[ParamSpec(name="pages_per_chunk", label="Pages per chunk", type="int", default=1, min=1, max=20),
            ParamSpec(name="max_tokens", label="Max tokens per chunk (0 = no limit)", type="int", default=800,
                      min=0, max=8000, unit="tokens"), TOKENIZER],
    inputs=["parsed pages"], outputs=["chunks with character spans"],
    compare_with=["structure_aware"],
    recommended_experiment="Compare page-level chunks with structure-aware on page-citation accuracy."))

registry.register(MethodSpec(
    id="structure_aware", category=C, stage=STAGE, name="Structure-aware (blocks, tables, captions)",
    summary="Cut at the document's own blocks and tables; keep each table whole with its caption.",
    how_it_works="Groups parsed blocks in reading order up to `size` tokens with block-level overlap. A table is "
                 "never merged with prose: it stays whole with its caption and Note/Source lines, a long table is "
                 "split by rows with the header repeated, and a table continued on the next page is linked. "
                 "Chunks under 50 tokens are folded into a neighbour; the section title is prefixed when embedding.",
    advantages=["Matches how the author organized the document", "Tables keep headers, captions and notes",
                "Every chunk carries pages, printed page and section"],
    disadvantages=["Needs good extraction", "Block-level overlap is uneven",
                   "Charts and diagrams contribute only their labels"],
    suitable_documents=["Long reports and handbooks with tables"], use_cases=["Cited retrieval over technical PDFs"],
    params=[size_param("tokens", 400, 50, 2000), overlap_param("tokens", 50, 1000)],
    inputs=["parsed pages (blocks and tables)"], outputs=["text chunks and table chunks"],
    requires=["PDF parsed with the PyMuPDF parser (tables detected)"],
    performance_notes="Near linear.", cost_notes="No model calls.",
    limitations=["Overlap is made of whole blocks, so it may be zero when blocks are larger than the overlap"],
    compare_with=["recursive", "paragraph", "page_aware"],
    recommended_experiment="Compare with Recursive on a PDF with tables, looking at table chunks and captions."))

registry.register(MethodSpec(
    id="table_rows", category=C, stage=STAGE, name="Table-aware (row-level)",
    summary="Prose as usual, but each table row becomes its own chunk that repeats the table caption and column headers.",
    how_it_works="Tables are split into one chunk per row (or `rows_per_chunk` rows). Each chunk carries the table caption, the header "
                 "row, and any Note/Source lines, either as a mini Markdown table or as 'Header: value' pairs. A table continued on the "
                 "next page keeps its caption. Prose uses the structure-aware chunker. A search for one cell then competes against "
                 "a single row instead of a whole grid.",
    advantages=["A question about one row or cell can match that row directly", "Headers and caption stay with every row",
                "Chunks stay small, so answers can quote them"],
    disadvantages=["Many more, smaller vectors for table-heavy documents", "A row on its own loses the comparison with its neighbours",
                   "Merged or blank header cells give weaker column labels"],
    suitable_documents=["Handbooks and reports with large reference tables", "Spec sheets, matrices"],
    use_cases=["Lookups such as 'what is the entrance criterion for X review'"],
    params=[size_param("tokens", 400, 50, 2000), overlap_param("tokens", 50, 1000),
            ParamSpec(name="rows_per_chunk", label="Rows per chunk", type="int", default=1, min=1, max=50, unit="rows",
                      help="1 = one row per chunk. Larger groups keep neighbouring rows together."),
            ParamSpec(name="row_format", label="Row format", type="select", default="markdown", options=["markdown", "key_value"],
                      help="markdown keeps a mini table; key_value writes 'Header: value; ...' which reads like a sentence.")],
    inputs=["parsed pages (blocks and tables)"], outputs=["text chunks and one chunk per table row"],
    requires=["a parser that detects tables (PyMuPDF, DOCX, HTML, CSV)"],
    performance_notes="Near linear; the vector count grows with the number of table rows.",
    cost_notes="More chunks to embed than whole-table chunking.",
    limitations=["Blank header cells (merged headers) become unlabeled columns in key_value format",
                 "Tables detected wrongly by the parser are split wrongly too"],
    compare_with=["structure_aware"],
    recommended_experiment="Index a table-heavy document with Structure-aware and with this method, then ask a question about one row and compare the top result."))

registry.register(MethodSpec(
    id="semantic", category=C, stage=STAGE, name="Semantic chunking",
    summary="Cut where the meaning shifts, found by embedding every sentence and comparing neighbours.",
    how_it_works="Splits text into sentences (tables stay whole), embeds each one, and measures how far each sentence is from the next. "
                 "A new chunk starts where that distance exceeds the chosen percentile of all distances, once the current chunk has "
                 "`min_tokens`; a chunk never grows past `max_tokens`. Chunks follow topic changes instead of a size budget.",
    advantages=["Chunks tend to be about one thing", "No fixed size to guess", "Works on any prose"],
    disadvantages=["Embeds every sentence (cost and time scale with the document)", "Chunk sizes vary a lot",
                   "Results depend on the embedding model and the percentile, which need tuning on your data"],
    suitable_documents=["Long prose with clear topic changes"], use_cases=["Articles, reports, transcripts"],
    params=[ParamSpec(name="provider", label="Embedding provider", type="select", default="azure_openai",
                      options=["azure_openai", "local_hash"], help="Used to embed sentences. local_hash is an offline demo that only matches shared words."),
            ParamSpec(name="breakpoint_percentile", label="Breakpoint percentile", type="int", default=90, min=50, max=99,
                      help="Cut at gaps larger than this percentile of all sentence-to-sentence distances. Higher = fewer, larger chunks."),
            ParamSpec(name="min_tokens", label="Minimum chunk size", type="int", default=60, min=1, max=2000, unit="tokens",
                      help="Do not cut before the chunk has at least this many tokens."),
            ParamSpec(name="max_tokens", label="Maximum chunk size", type="int", default=500, min=20, max=8000, unit="tokens",
                      help="Force a cut when a chunk would exceed this."), TOKENIZER],
    inputs=["canonical text"], outputs=["chunks with character spans"], requires=["an embedding provider"],
    performance_notes="One embedding call per ~64 sentences; results are cached, so re-running costs nothing.",
    cost_notes="Embedding tokens for the whole document each time the text changes.",
    limitations=["Sentence splitting is rule-based", "min_tokens must not exceed max_tokens", "Not meaningful with the offline demo provider"],
    compare_with=["recursive", "sentence", "structure_aware"],
    recommended_experiment="Chunk a long report semantically and with Recursive at a similar size, index both, and compare them in the Evaluation Center."))

registry.register(MethodSpec(
    id="parent_child", category=C, stage=STAGE, name="Parent-child (small-to-big)",
    summary="Search small chunks for precision, then give the model the larger passage around each match.",
    how_it_works="The text is split into large parent passages (recursive splitting); each parent is divided into small child chunks "
                 "with a token window. Only children are embedded, so a query matches a narrow span. Each child remembers its parent. "
                 "At search time, choose 'expand to parent' to return the whole parent instead of the child; several children of one "
                 "parent collapse into a single result.",
    advantages=["Precise matching with full context in the answer prompt", "One parent returned even if many children match"],
    disadvantages=["Two levels to store and reason about", "Parents are bigger, so answers spend more prompt tokens",
                   "Children are blind to anything outside their window when matching"],
    suitable_documents=["Long prose where answers need surrounding context"], use_cases=["Q&A over reports and handbooks"],
    params=[ParamSpec(name="parent_size", label="Parent size", type="int", default=1200, min=20, max=8000, unit="tokens",
                      help="Largest passage returned when expanding to the parent."),
            ParamSpec(name="child_size", label="Child size", type="int", default=200, min=20, max=2000, unit="tokens",
                      help="Size of the small chunks that are embedded and searched. Must be smaller than the parent."),
            ParamSpec(name="child_overlap", label="Child overlap", type="int", default=20, min=0, max=1000, unit="tokens",
                      help="Overlap between neighbouring children. Must be smaller than the child size."), TOKENIZER],
    inputs=["canonical text"], outputs=["child chunks with a parent reference"],
    performance_notes="More, smaller vectors than a single-level chunker.", cost_notes="Embedding cost follows the children only.",
    limitations=["Parents come from recursive splitting, which ignores tables and layout", "Page citations use the child's pages"],
    compare_with=["recursive", "structure_aware"],
    recommended_experiment="Index with this method, search with and without 'expand to parent', and compare answers on questions that need context."))

# ---------------------------------------------------------------- planned (not executable yet)
_planned("overlapping", "Overlapping chunking (preset)",
         "Any size-based method with overlap > 0.",
         "Not a separate algorithm: it is the `overlap` setting of fixed, recursive, sentence, paragraph and "
         "sliding-window methods.",
         ["Protects facts near boundaries"], ["Duplicates text, grows the index"],
         "Covered by the overlap parameter of the implemented methods; no separate method needed.")
_planned("non_overlapping", "Non-overlapping chunking (preset)", "Any size-based method with overlap = 0.",
         "Not a separate algorithm: set overlap to 0.", ["Smallest index"], ["Facts can be cut at boundaries"],
         "Covered by setting overlap to 0 on the implemented methods.")
_planned("html_structure", "HTML structure-based", "Cut at HTML headings, lists and tables.",
         "Walk the DOM and split at structural tags, keeping tables whole.",
         ["Follows the page's real structure"], ["Needs clean HTML"],
         "Planned (Stage 3). HTML files are parsed to text today, but a DOM-aware splitter is not built.")
_planned("section_based", "Document section-based", "One chunk per detected section.",
         "Uses detected section titles from the parser as boundaries.",
         ["Chunks match sections"], ["Sections vary wildly in size"],
         "Planned (Stage 3). Section titles are detected for numbered PDF headings, but section chunking is not built.")
_planned("topic_based", "Topic-based", "Cut at topic changes found by clustering or a topic model.",
         "Cluster sentence embeddings or run a topic segmentation model.",
         ["Topic-coherent chunks"], ["Heavy, unstable on short texts"],
         "Planned (Stage 3). Not built.", requires=["embedding model"], creds=True)
_planned("layout_aware", "Layout-aware", "Use reading order, columns and block types from a layout model.",
         "Needs a layout-analysis parser (Docling, Unstructured, cloud document AI).",
         ["Handles multi-column and complex pages"], ["Needs a layout parser"],
         "Planned (Stage 5). Only the PyMuPDF parser exists.", requires=["layout parser"])
_planned("code_aware", "Code-aware (tree-sitter)", "Cut at functions and classes using a syntax parser.",
         "Parse source with tree-sitter and split at definition boundaries.",
         ["Chunks are whole functions"], ["Per-language grammars"],
         "Planned (Stage 5). Source code files are not yet an accepted upload type.", requires=["tree-sitter"])
_planned("json_xml_structure", "JSON/XML structure-aware", "Cut at records, elements or array items.",
         "Walk the tree and emit one chunk per record or subtree.",
         ["Keeps records intact"], ["Needs schema awareness"],
         "Planned (Stage 5). JSON is ingested as text; XML is not yet accepted.")
_planned("contextual", "Contextual chunk enrichment", "An LLM writes a short 'where this fits' note for each chunk.",
         "For each chunk, send the document (or a window) and the chunk to an LLM and prepend its one-sentence "
         "context before embedding.",
         ["Often a large retrieval gain"], ["LLM call per chunk; cost scales with corpus size"],
         "Planned (Stage 3). Needs the generation service wired in; will run on Azure OpenAI.",
         requires=["LLM"], creds=True)
_planned("proposition", "LLM proposition / atomic-fact chunking",
         "An LLM rewrites text into self-contained factual statements.",
         "Prompt an LLM to extract standalone propositions, then embed each one.",
         ["Very precise retrieval units"], ["Expensive; may paraphrase or hallucinate"],
         "Planned (Stage 3). Needs the generation service.", requires=["LLM"], creds=True)
_planned("adaptive", "Adaptive (by document structure)", "Choose the chunking method per document.",
         "Inspect each document (tables, headings, length) and pick a method.",
         ["No one-size-fits-all choice"], ["Selection rules need evaluation"],
         "Planned (Stage 6), after evaluation exists to justify the rules.")

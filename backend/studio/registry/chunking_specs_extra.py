"""Registry entries for the remaining chunking methods (implemented in chunking/methods_extra.py)."""
from .base import MethodSpec, ParamSpec, registry
from .chunking_specs import C, STAGE, TOKENIZER, size_param

MIN = lambda d=40: ParamSpec(name="min_tokens", label="Minimum chunk size (tokens)", type="int", default=d, min=0, max=2000, unit="tokens",  # noqa: E731
                             help="Chunks smaller than this are merged into a neighbour unless a hard boundary separates them.")


def reg(**kw) -> None:
    registry.register(MethodSpec(category=C, stage=STAGE, inputs=kw.pop("inputs", ["canonical text"]),
                                 outputs=kw.pop("outputs", ["chunks with character spans"]), **kw))


reg(id="overlapping", name="Overlapping (preset)",
    summary="Fixed token windows where each chunk repeats a percentage of the previous one.",
    how_it_works="Cuts the token stream into windows of `size` tokens that advance by size minus the overlap, so a fact that straddles a "
                 "boundary is whole in at least one chunk. The overlap is given as a percentage of the size.",
    advantages=["Facts near a boundary survive", "Simple and predictable"], disadvantages=["Costs extra embedding tokens (about 1/(1-overlap) times the text)", "Cuts mid-sentence"],
    suitable_documents=["Any text"], use_cases=["Baseline with a safety margin"],
    params=[size_param("tokens", 400, 20, 8000),
            ParamSpec(name="overlap_percent", label="Overlap (%)", type="int", default=15, min=1, max=90, unit="%",
                      help="Share of each chunk repeated at the start of the next."), TOKENIZER],
    compare_with=["non_overlapping", "fixed_token"], recommended_experiment="Run 0%, 10%, 20% and 40% overlap and compare recall against embedding cost.")

reg(id="non_overlapping", name="Non-overlapping (preset)",
    summary="Chunks that never repeat text, cut cleanly by paragraph and sentence or by fixed token count.",
    how_it_works="Either recursive splitting with overlap forced to zero (clean breaks at paragraphs, lines and sentences) or fixed token windows "
                 "of exactly `size` tokens.",
    advantages=["Cheapest: embedding cost equals the document", "No duplicate hits for one passage"], disadvantages=["A fact on a boundary can be split across two chunks"],
    suitable_documents=["Any text"], use_cases=["Cost-sensitive corpora"],
    params=[size_param("tokens", 400, 20, 8000),
            ParamSpec(name="strategy", label="Strategy", type="select", default="recursive", options=["recursive", "fixed_token"],
                      help="Recursive respects paragraph and sentence breaks; fixed_token cuts exactly every size tokens."), TOKENIZER],
    compare_with=["overlapping", "recursive"], recommended_experiment="Compare against the overlapping preset at the same size on your questions.")

reg(id="html_structure", name="HTML structure-based",
    summary="Cut at HTML headings, keep lists together and tables whole.",
    how_it_works="Uses the element names kept by the HTML parser: a heading (h1 to h6) starts a new chunk, consecutive list items stay together, "
                 "each table is its own chunk, and paragraphs fill the rest up to the size limit.",
    advantages=["Chunks follow the page's own structure", "Tables and lists are never cut mid-way"], disadvantages=["Only HTML uploads carry element names", "Navigation and boilerplate are removed by the parser, not here"],
    suitable_documents=["Saved web pages", "HTML documentation"], use_cases=["Docs sites and knowledge-base pages"],
    params=[size_param("tokens", 500, 50, 8000), MIN(60),
            ParamSpec(name="split_on_headings", label="Start a chunk at each heading", type="bool", default=True),
            ParamSpec(name="keep_lists_together", label="Keep list items together", type="bool", default=True), TOKENIZER],
    limitations=["Falls back to recursive splitting for non-HTML files, and says so"], compare_with=["markdown_header", "section_based"],
    recommended_experiment="Upload a saved documentation page and compare with Recursive on a question about a list.")

reg(id="section_based", name="Document section-based",
    summary="One chunk per section, found from headings and the parser's section titles.",
    how_it_works="A new section starts where the parser's section title changes or a line looks like a heading (Markdown #, numbered '3.2 Title', "
                 "'Appendix A', short ALL-CAPS lines). Sections are packed up to the size limit, tiny sections merge forward, long sections are cut at sentences.",
    advantages=["Each chunk is about one subject", "Section title is a natural label for citations"], disadvantages=["Sections vary widely in size", "Heading detection is rule-based and can miss unusual styles"],
    suitable_documents=["Handbooks, reports, specifications, manuals"], use_cases=["Long structured documents"],
    params=[size_param("tokens", 800, 50, 8000), MIN(60), TOKENIZER],
    compare_with=["layout_aware", "structure_aware", "markdown_header"], recommended_experiment="Compare with Structure-aware on a handbook; check how often a question's answer needs two sections.")

reg(id="topic_based", name="Topic-based (TextTiling)",
    summary="Cut where the vocabulary of the text changes, found without any model.",
    how_it_works="TextTiling: for every gap between sentences, compare the words of the k sentences before with the k after (cosine similarity). "
                 "A deep valley in that similarity is a topic change. Boundaries are the valleys deeper than the mean plus a sensitivity times the standard deviation; "
                 "chunks are then kept within the minimum and maximum size.",
    advantages=["Free, deterministic, no embedding calls", "Finds topic shifts that headings do not show"], disadvantages=["Lexical only: synonyms look like a change", "Noisy on very short sentences"],
    suitable_documents=["Long prose without headings: transcripts, articles, notes"], use_cases=["Unstructured long text"],
    params=[ParamSpec(name="block_sentences", label="Block size (sentences)", type="int", default=6, min=2, max=30, unit="sentences",
                      help="Sentences compared on each side of a gap."),
            ParamSpec(name="sensitivity", label="Sensitivity", type="float", default=0.5, min=-1, max=3, step=0.1,
                      help="Higher means fewer, stronger boundaries."), MIN(80),
            ParamSpec(name="max_tokens", label="Maximum chunk size (tokens)", type="int", default=600, min=50, max=8000, unit="tokens"), TOKENIZER],
    compare_with=["semantic", "paragraph"], recommended_experiment="Compare with Semantic chunking: same idea, lexical versus embedding-based, one free and one paid.")

reg(id="layout_aware", name="Layout-aware",
    summary="Use the page layout (font size, columns, gaps) to find headings and blocks that belong together.",
    how_it_works="PDF text blocks carry their position and font size. A block set in a clearly larger font starts a new chunk (a heading), a block that starts "
                 "to the right of and above the previous one is a new column, and a large vertical gap is a new block of content. Tables are separate. "
                 "Needs layout data, which is recorded for PDFs uploaded after this method existed.",
    advantages=["Works on PDFs that have no heading text markers", "Respects columns"], disadvantages=["PDF only", "Documents ingested earlier must be re-uploaded", "Font-size heuristics can misread decorative text"],
    suitable_documents=["Reports, papers, brochures with headings set by font size"], use_cases=["Typeset PDFs"],
    params=[size_param("tokens", 600, 50, 8000), MIN(60),
            ParamSpec(name="heading_ratio", label="Heading font ratio", type="float", default=1.15, min=1.0, max=3.0, step=0.05,
                      help="A block counts as a heading when its font is at least this many times the body font."),
            ParamSpec(name="gap_factor", label="Gap factor", type="float", default=2.0, min=0.5, max=10, step=0.5,
                      help="A vertical gap larger than this many text-line heights starts a new chunk."),
            ParamSpec(name="split_pages", label="Always split at page breaks", type="bool", default=False), TOKENIZER],
    limitations=["Falls back to recursive splitting when the document has no layout data, and says so"], compare_with=["section_based", "structure_aware"],
    recommended_experiment="Compare with Section-based on a two-column paper.")

reg(id="code_aware", name="Code-aware",
    summary="Cut source code at functions and classes instead of at arbitrary lines.",
    how_it_works="Python is parsed into a syntax tree, so each top-level function and class (with its decorators and leading comments) is a unit and "
                 "runs of imports or assignments form another. Other languages use declaration patterns at the start of a line "
                 "(function, class, fn, func, struct, impl ...). Small neighbours merge up to the size limit; a function longer than the limit is cut at lines and "
                 "can repeat its signature line.",
    advantages=["A function is never split from its signature", "Docstrings and decorators stay with their code"], disadvantages=["Pattern-based splitting for non-Python languages is approximate", "Deeply nested code is cut only at the top level"],
    suitable_documents=["Source files"], use_cases=["Code search and repository question answering"],
    params=[size_param("tokens", 500, 50, 8000), MIN(80),
            ParamSpec(name="language", label="Language", type="select", default="auto", options=["auto", "python", "other"],
                      help="Auto tries Python's parser first, then declaration patterns."),
            ParamSpec(name="merge_small", label="Merge small definitions", type="bool", default=True),
            ParamSpec(name="repeat_signature", label="Repeat the signature in long functions", type="bool", default=True), TOKENIZER],
    requires=["python ast (standard library)"], limitations=["Not tree-sitter: only Python has a real syntax tree"], compare_with=["recursive", "fixed_token"],
    recommended_experiment="Upload a .py file and compare with Fixed-size on a question that names a function.")

reg(id="json_xml_structure", name="JSON/XML structure-aware",
    summary="Cut at records: one JSON array item, object key or XML element per chunk (grouped up to the size limit).",
    how_it_works="For JSON the records are the direct children of the top-level array or object (descending one level into a single-key wrapper such as "
                 "{\"items\": [...]}), located with the JSON decoder so each chunk is an exact slice. For XML (parsed to 'path: text' lines) a record starts "
                 "where the first path of the record repeats. Records pack up to the size limit; an oversized record is cut recursively.",
    advantages=["Every record stays whole and valid", "Exact spans"], disadvantages=["Needs repeating records; a single deeply nested object is cut by size"],
    suitable_documents=["JSON exports, API responses, XML data feeds"], use_cases=["Catalogues, logs, product feeds"],
    params=[size_param("tokens", 400, 20, 8000),
            ParamSpec(name="records_per_chunk", label="Records per chunk", type="int", default=1, min=1, max=1000,
                      help="1 packs by size; a larger number forces exactly that many records per chunk."),
            ParamSpec(name="pack", label="Pack small records up to the size limit", type="bool", default=True), TOKENIZER],
    limitations=["Falls back to recursive splitting when no repeating records exist, and says so"], compare_with=["recursive", "table_rows"],
    recommended_experiment="Upload a JSON array of 200 records and compare with Fixed-size on a question about one record.")

reg(id="contextual", status="requires_credentials", name="Contextual chunking (LLM)",
    summary="Cut like Recursive, then a chat model writes one sentence per chunk that places it in the document.",
    how_it_works="Chunks are made by recursive splitting. For each chunk the model sees the file name, section title, 500 characters before and after and the chunk, "
                 "and writes one sentence of context (at most 35 words, clipped to 320 characters). The sentence is stored beside the chunk, never merged into its text, "
                 "and is embedded with the chunk when an index is built from this method.",
    advantages=["Chunks that say 'it' or 'the system' become findable", "The added sentence is visible and auditable"], disadvantages=["One chat call per chunk", "The model can only restate what is nearby"],
    suitable_documents=["Documents where chunks lose their subject when cut"], use_cases=["Technical handbooks with many cross-references"],
    params=[size_param("tokens", 400, 50, 4000),
            ParamSpec(name="overlap", label="Overlap (tokens)", type="int", default=40, min=0, max=500, unit="tokens"),
            ParamSpec(name="max_chunks", label="Chunks to enrich", type="int", default=25, min=0, max=100000,
                      help="Spend control. 0 enriches every chunk (one chat call each). Set it to 0 when building an index."),
            ParamSpec(name="model", label="Chat deployment (blank = default)", type="text", default=""), TOKENIZER],
    requires=["Azure OpenAI chat deployment"], needs_cloud_credentials=True, cost_notes="One chat call (about 400 tokens in, 40 out) per enriched chunk.",
    compare_with=["recursive", "parent_child"], recommended_experiment="Build two indexes (with and without context) and run the evaluation grid on both.")

reg(id="proposition", status="requires_credentials", name="Proposition / atomic-fact chunking (LLM)",
    summary="A chat model rewrites each passage as short, self-contained facts; each fact becomes a chunk.",
    how_it_works="Passages are cut by recursive splitting. The model rewrites each into standalone statements with names instead of pronouns. A statement is kept only "
                 "if at least the chosen share of its content words occur in its source passage (a guard against invention). Chunks are the model's wording, so they have no "
                 "highlight in the document, but they keep the source pages and section for citations.",
    advantages=["Very precise retrieval units", "Each unit makes sense alone"], disadvantages=["One chat call per passage", "Paraphrase can drift from the source wording", "Loses exact spans"],
    suitable_documents=["Dense factual prose"], use_cases=["Fact lookup questions"],
    params=[ParamSpec(name="source_size", label="Source passage size (tokens)", type="int", default=300, min=50, max=2000, unit="tokens"),
            ParamSpec(name="per_chunk", label="Statements per chunk", type="int", default=1, min=1, max=10),
            ParamSpec(name="min_overlap", label="Minimum word overlap with source", type="float", default=0.7, min=0.3, max=1.0, step=0.05,
                      help="Share of a statement's content words that must occur in the source passage."),
            ParamSpec(name="max_passages", label="Passages to rewrite", type="int", default=10, min=0, max=100000,
                      help="Spend control. 0 rewrites every passage (one chat call each)."),
            ParamSpec(name="model", label="Chat deployment (blank = default)", type="text", default=""), TOKENIZER],
    requires=["Azure OpenAI chat deployment"], needs_cloud_credentials=True, cost_notes="One chat call per source passage.",
    compare_with=["sentence", "contextual"], recommended_experiment="Compare with Sentence-based on fact-lookup questions.")

reg(id="adaptive", name="Adaptive (by document structure)",
    summary="Measures the document and picks the chunking method that suits it, then explains the choice.",
    how_it_works="Rules, in order: source code -> Code-aware; JSON/XML records -> JSON/XML structure; HTML elements -> HTML structure; 3+ Markdown headings -> Markdown header; "
                 "tables over 20% of the text -> Structure-aware; 5+ larger-font headings -> Layout-aware; clear sections -> Section-based; many short blocks -> Paragraph; "
                 "otherwise Recursive. The measurements and the reason are printed with the result.",
    advantages=["No need to know the document type in advance", "The decision is explained, not hidden"], disadvantages=["The rules are heuristics: check them with the evaluation grid"],
    suitable_documents=["Mixed collections"], use_cases=["First pass on an unknown corpus"],
    params=[size_param("tokens", 500, 50, 8000, "Target size passed to the chosen method when it has one.")],
    compare_with=["recursive", "section_based"], recommended_experiment="Run on every document type you have and read the explanation for each.")

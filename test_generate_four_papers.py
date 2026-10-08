"""
Multi-Format Academic Typesetting Test
──────────────────────────────────────
Generates four distinct publication-grade manuscripts from the identical research content:
  1. ResearchAgent_IEEE.pdf  (Two-column, IEEEtran geometry, Roman section numbering, IEEE CSL)
  2. ResearchAgent_ACM.pdf   (Two-column, ACM sigconf, Sans headers, CCS concepts, ACM CSL)
  3. ResearchAgent_APA.pdf   (Single-column, APA 7th title page, ragged right, Author-Date CSL)
  4. ResearchAgent_MLA.pdf   (Single-column, MLA 9th header, double spaced, Author-Page CSL)
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.pdf_export import generate_research_pdf

TOPIC = "Retrieval-Augmented Generation for Code Generation"

MANUSCRIPT_MARKDOWN = r"""# Retrieval-Augmented Generation for Code Generation: Structural AST Grounding and Empirical Synthesis

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a cornerstone paradigm for mitigating hallucination and updating outdated parametric knowledge in Large Language Models (LLMs) deployed for software engineering. Standard text-based retrieval, however, disregards the rich hierarchical syntax, control flow, and dependency graphs inherent to software source code. In this paper, we synthesize recent advances in Code-RAG, establishing a unified formulation that bridges lexical retrieval, dense semantic vector search, and Abstract Syntax Tree (AST) sub-graph retrieval. We rigorously evaluate how structural conditioning impacts Pass@k metrics on canonical benchmarks and formalize the compound loss governing retrieval-aware code completion. Finally, we categorize open research challenges including repository-level context fragmentation, test-suite execution feedback loops, and semantic patch validation.

## Keywords
code generation, retrieval-augmented generation, abstract syntax trees, software synthesis, dense retrieval, static verification

## 1. Introduction
The advent of specialized foundation models for code, such as CodeX, StarCoder, and Code Llama, has transformed automated software engineering. Despite impressive fluency, generative code models suffer from chronic limitations: parametric staleness when libraries evolve, hallucination of non-existent API methods, and an inability to perceive private repository-level dependencies [@lewis2020rag].

Retrieval-Augmented Generation addresses these deficiencies by dynamically retrieving relevant context snippets from external codebases prior to token generation [@chen2021codex]. However, treating source code as flat natural language text overlooks its structural duality: code is simultaneously a sequence of tokens and a strictly typed Abstract Syntax Tree (AST) governed by formal grammar [@feng2020codebert].

## 2. Theoretical Framework and Problem Formulation
Let $\mathcal{C}$ denote an expansive software repository comprising source files, class definitions, and documentation. When presented with a programmatic query $q$ (such as an incomplete function signature or docstring), the retrieval objective is to sample an optimal context set $\mathcal{D}^* \subset \mathcal{C}$ that maximizes generation fidelity under model parameters $\theta$:

$$P(y \mid q, \mathcal{C}) = \sum_{d \in \mathcal{D}^*} P(d \mid q) \cdot \prod_{t=1}^T P(y_t \mid y_{<t}, q, d; \theta)$$

### 2.1 Joint AST and Dense Hybrid Scoring
To bridge textual semantics and grammar constraints, modern systems utilize hybrid retrieval. The combined similarity score $S(q, d)$ balances dense dual-encoder embeddings with AST tree edit distances:

$$S(q, d) = \lambda \cdot \frac{\mathbf{e}_q^\top \mathbf{e}_d}{\|\mathbf{e}_q\| \|\mathbf{e}_d\|} + (1 - \lambda) \cdot \exp\left( -\gamma \cdot \operatorname{TEDS}(T_q, T_d) \right)$$

where $T_q$ and $T_d$ represent parsed AST sub-trees, $\operatorname{TEDS}(\cdot)$ denotes Tree Edit Distance Similarity, and $\lambda \in [0, 1]$ represents the learned gating coefficient.

## 3. Empirical Benchmark Analysis
We synthesize experimental outcomes across HumanEval-Repo and SWE-bench to evaluate the delta between naive BM25, dense neural retrieval (Dense-RAG), and AST-informed hybrid retrieval.

| Architecture | Retrieval Mode | Context Window | HumanEval Pass@1 | SWE-bench Resolve % | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Standard LLM (Zero-RAG) | None | 4,096 | 48.2% | 4.8% | 142 |
| BM25 Lexical Baseline | Lexical Tokens | 8,192 | 56.4% | 11.2% | 198 |
| Dense Bi-Encoder | MiniLM-L6 | 8,192 | 64.7% | 18.5% | 230 |
| GraphCodeBERT + AST | Sub-graph GNN | 16,384 | 71.3% | 24.1% | 315 |
| ResearchAgent Hybrid | Multi-Vector AST | 16,384 | 78.6% | 29.8% | 340 |

As detailed in Table 1, augmenting parametric models with structural AST grounding yields a +22.2% absolute gain in HumanEval Pass@1 compared to lexical retrieval alone.

## 4. Discussion and Limitations
While retrieval-augmented generation substantially reduces syntactic hallucinations, several architectural bottlenecks remain:
- **Context Window Fragmentation:** Naive chunking breaks lexical scopes, leading to orphan variable references.
- **Cross-File Import Resolution:** Dynamic runtime imports in languages such as Python and JavaScript defy static resolution without active language server protocol (LSP) indexing.
- **Execution-Guided Verification:** Static generation must be coupled with automated unit test generation and compiler feedback to ensure generated code passes sandboxed validation.

## 5. Conclusion
This synthesis demonstrates that source-code retrieval requires fundamentally different inductive biases than general-domain text search. By harmonizing dense vector representations with AST topological structures, code synthesis engines attain unprecedented precision and real-world repository comprehension. Future research must prioritize incremental repository index updates and multi-modal execution trace grounding.
"""

BIBLIOGRAPHY = [
    {
        "paper_id": "lewis2020rag",
        "citation_key": "lewis2020rag",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin",
        "year": "2020",
        "source": "Advances in Neural Information Processing Systems (NeurIPS)",
        "volume": "33",
        "pages": "9459--9474",
        "doi": "10.5555/3495724.3496517",
        "verified": True,
    },
    {
        "paper_id": "chen2021codex",
        "citation_key": "chen2021codex",
        "title": "Evaluating Large Language Models Trained on Code",
        "authors": "Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde",
        "year": "2021",
        "source": "arXiv preprint arXiv:2107.03374",
        "url": "https://arxiv.org/abs/2107.03374",
        "verified": True,
    },
    {
        "paper_id": "feng2020codebert",
        "citation_key": "feng2020codebert",
        "title": "CodeBERT: A Pre-Trained Model for Programming and Natural Languages",
        "authors": "Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng",
        "year": "2020",
        "source": "Findings of the Association for Computational Linguistics: EMNLP 2020",
        "pages": "1536--1547",
        "doi": "10.18653/v1/2020.findings-emnlp.139",
        "verified": True,
    },
    {
        "paper_id": "vaswani2017attention",
        "citation_key": "vaswani2017attention",
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit",
        "year": "2017",
        "source": "Advances in Neural Information Processing Systems (NeurIPS)",
        "volume": "30",
        "pages": "5998--6008",
        "verified": True,
    },
]


def run_test():
    out_dir = Path(__file__).parent / "generated_test_papers"
    out_dir.mkdir(parents=True, exist_ok=True)

    formats = ["ieee", "acm", "apa", "mla"]
    results = {}

    for fmt in formats:
        pdf_name = f"ResearchAgent_{fmt.upper()}.pdf"
        out_pdf = out_dir / pdf_name
        res_dir = out_dir / f"res_{fmt}"
        res_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n==========================================")
        print(f"Generating {fmt.upper()} Manuscript...")
        print(f"Target: {out_pdf}")

        generated = generate_research_pdf(
            topic=TOPIC,
            session_id=f"test_{fmt}",
            analysis_markdown="",
            final_markdown=MANUSCRIPT_MARKDOWN,
            out_path=out_pdf,
            resource_dir=res_dir,
            paper_format=fmt,
            bibliography_data=BIBLIOGRAPHY,
        )

        size_kb = generated.stat().st_size / 1024.0
        print(f"SUCCESS: {pdf_name} generated ({size_kb:.1f} KB)")
        results[fmt] = {
            "path": str(generated),
            "size_kb": size_kb,
        }

    print("\n==========================================")
    print("ALL 4 MANUSCRIPTS GENERATED SUCCESSFULLY:")
    for fmt, r in results.items():
        print(f"  - [{fmt.upper()}]: {r['size_kb']:.1f} KB -> {r['path']}")


if __name__ == "__main__":
    run_test()

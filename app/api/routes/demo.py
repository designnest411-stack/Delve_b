"""Demo paper endpoint — serves a pre-generated research paper without authentication."""

from fastapi import APIRouter

router = APIRouter(prefix="/demo", tags=["demo"])

# Pre-generated demo paper (a real example you've already created)
DEMO_PAPER = {
    "topic": "Advances in Transformer Architecture Efficiency",
    "paper": """# Advances in Transformer Architecture Efficiency: A Comprehensive Review

## Abstract

This paper reviews recent advances in making transformer architectures more computationally efficient while maintaining performance. We examine sparse attention mechanisms, knowledge distillation, quantization techniques, and architectural innovations that reduce memory footprint and inference latency.

## 1. Introduction

Transformer models have revolutionized natural language processing and computer vision, but their quadratic complexity with respect to sequence length poses significant computational challenges. This review synthesizes recent research addressing these efficiency bottlenecks.

## 2. Sparse Attention Mechanisms

### 2.1 Locality-Sensitive Hashing

Reformer (Kitaev et al., 2020) introduced locality-sensitive hashing to reduce attention complexity from O(n²) to O(n log n)...

### 2.2 Fixed Patterns

Longformer (Beltagy et al., 2020) combines local windowed attention with task-specific global attention patterns...

## 3. Knowledge Distillation

DistilBERT (Sanh et al., 2019) demonstrated that student models can retain 97% of teacher performance with 40% fewer parameters...

## 4. Quantization Approaches

### 4.1 Post-Training Quantization

Recent work shows INT8 quantization can maintain accuracy while reducing model size by 4×...

## 5. Architectural Innovations

### 5.1 Mixture of Experts

Switch Transformer (Fedus et al., 2021) scales to trillion-parameter models by activating only a subset of parameters per token...

## 6. Cross-Paper Analysis

A recurring theme is the trade-off between efficiency and few-shot learning capability. Models optimized for inference speed often struggle with rapid adaptation...

## 7. Research Gaps

### Gap 1: Long-context efficiency under 10K tokens
Most efficiency work targets sequences >10K tokens, leaving mid-range applications under-optimized...

### Gap 2: Hardware-aware architecture search
Current methods optimize FLOPs but ignore memory bandwidth constraints on modern accelerators...

## 8. Conclusion

Transformer efficiency remains an active research area. Future work should focus on hardware co-design and task-specific optimizations.

## References

1. Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). Reformer: The Efficient Transformer. ICLR.
2. Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. arXiv.
3. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. NeurIPS Workshop.
4. Fedus, W., et al. (2021). Switch Transformers: Scaling to Trillion Parameter Models. JMLR.
""",
    "final_draft": """# Advances in Transformer Architecture Efficiency: A Comprehensive Review

## Abstract

This paper reviews recent advances in making transformer architectures more computationally efficient...

[Same content as above - this would be the cleaned version without analysis]
""",
    "analysis": """## Research Quality Assessment

**Papers Reviewed:** 18 peer-reviewed papers (2019-2024)
**Average Citation Count:** 287
**Source Distribution:**
- arXiv: 8 papers
- Conference proceedings: 10 papers

**Key Findings:**
- Sparse attention reduces complexity by 60-80%
- Knowledge distillation preserves 95%+ accuracy
- Quantization achieves 4× compression with <2% degradation
""",
    "bibliography": [
        {
            "title": "Reformer: The Efficient Transformer",
            "authors": ["Nikita Kitaev", "Łukasz Kaiser", "Anselm Levskaya"],
            "year": 2020,
            "source": "arxiv",
            "citation_count": 1243
        },
        {
            "title": "Longformer: The Long-Document Transformer",
            "authors": ["Iz Beltagy", "Matthew E. Peters", "Arman Cohan"],
            "year": 2020,
            "source": "arxiv",
            "citation_count": 892
        },
        {
            "title": "DistilBERT, a distilled version of BERT",
            "authors": ["Victor Sanh", "Lysandre Debut", "Julien Chaumond", "Thomas Wolf"],
            "year": 2019,
            "source": "arxiv",
            "citation_count": 2456
        },
    ],
    "gaps": [
        {
            "title": "Long-context efficiency under 10K tokens",
            "evidence": "Most efficiency work targets sequences >10K tokens, leaving mid-range applications under-optimized",
            "proposed_direction": "Develop hybrid attention mechanisms optimized for the 1K-10K token range commonly seen in production NLP"
        },
        {
            "title": "Hardware-aware architecture search",
            "evidence": "Current methods optimize FLOPs but ignore memory bandwidth constraints on modern accelerators",
            "proposed_direction": "Co-design transformers with hardware profiling to minimize memory access patterns"
        }
    ],
    "metrics": {
        "papers_reviewed": 18,
        "avg_citation_count": 287,
        "elapsed_seconds": 142,
        "token_estimate": 0,
        "cost_estimate_usd": 0
    }
}


@router.get("/paper")
async def get_demo_paper() -> dict:
    """Returns a pre-generated demo paper. No authentication required."""
    return DEMO_PAPER

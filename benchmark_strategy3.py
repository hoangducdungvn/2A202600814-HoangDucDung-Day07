# -*- coding: utf-8 -*-
"""
Strategy 3 — RecursiveChunker Benchmark
========================================
Thử separators theo thứ tự: \n\n → \n → . → " " → ""
So sánh với baseline (FixedSizeChunker chunk_size=500, overlap=50)

Mục tiêu:
- Chạy ChunkingStrategyComparator trên tất cả tài liệu VinFast
- Tuning RecursiveChunker với nhiều chunk_size khác nhau
- Chạy 5 benchmark queries qua EmbeddingStore + KnowledgeBaseAgent
- So sánh kết quả với baseline FixedSizeChunker
"""

import sys
import io
import os
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src import (
    Document,
    FixedSizeChunker,
    RecursiveChunker,
    ChunkingStrategyComparator,
    EmbeddingStore,
    KnowledgeBaseAgent,
    compute_similarity,
    _mock_embed,
)

# ============================================================
# 1. Load tất cả tài liệu VinFast
# ============================================================
DATA_DIR = Path("data")
VINFAST_FILES = [
    ("baomatcanhan.md", "privacy", "Chính sách Bảo vệ Dữ liệu Cá nhân"),
    ("chinhsachbaomat.md", "security", "Chính sách Bảo mật"),
    ("chinhsachthuepin.md", "battery_rental", "Chính sách Dịch vụ Cho thuê Pin"),
    ("chinh_sach_san_pham.md", "product", "Chính sách Sản phẩm"),
    ("dieukhoan.md", "terms", "Điều khoản và Điều kiện Giao dịch"),
    ("dieukhoandatcoc.md", "deposit", "Điều khoản Đặt cọc Mua xe"),
    ("dieukiensudung.md", "usage", "Điều kiện Sử dụng Website"),
]


def load_vinfast_docs():
    """Load tất cả tài liệu VinFast với metadata."""
    docs = []
    for filename, category, title in VINFAST_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"  [SKIP] {filename} not found")
            continue
        content = filepath.read_text(encoding="utf-8")
        docs.append(Document(
            id=filepath.stem,
            content=content,
            metadata={
                "source": str(filepath),
                "category": category,
                "title": title,
                "language": "vi",
                "extension": filepath.suffix,
            },
        ))
    return docs


# ============================================================
# 2. So sánh 3 strategies (baseline comparison)
# ============================================================
def run_strategy_comparison(docs):
    print("=" * 70)
    print("PHẦN 1: SO SÁNH 3 CHUNKING STRATEGIES (ChunkingStrategyComparator)")
    print("=" * 70)

    comparator = ChunkingStrategyComparator()
    for doc in docs:
        print(f"\n--- {doc.metadata['title']} ({len(doc.content)} chars) ---")
        result = comparator.compare(doc.content, chunk_size=500)
        for strategy, stats in result.items():
            print(f"  {strategy:15s}: count={stats['count']:3d}, avg_length={stats['avg_length']:.0f}")


# ============================================================
# 3. RecursiveChunker parameter tuning
# ============================================================
def run_parameter_tuning(docs):
    print("\n" + "=" * 70)
    print("PHẦN 2: TUNING RecursiveChunker — THỬ NHIỀU chunk_size")
    print("=" * 70)

    for doc in docs:
        print(f"\n--- {doc.metadata['title']} ({len(doc.content)} chars) ---")
        print(f"  {'chunk_size':>10s} | {'count':>5s} | {'avg':>5s} | {'min':>5s} | {'max':>5s}")
        print(f"  {'-'*10} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*5}")
        for cs in [200, 300, 500, 800, 1000]:
            rc = RecursiveChunker(chunk_size=cs)
            chunks = rc.chunk(doc.content)
            if not chunks:
                continue
            avg = sum(len(c) for c in chunks) / len(chunks)
            min_len = min(len(c) for c in chunks)
            max_len = max(len(c) for c in chunks)
            print(f"  {cs:>10d} | {len(chunks):>5d} | {avg:>5.0f} | {min_len:>5d} | {max_len:>5d}")


# ============================================================
# 4. So sánh RecursiveChunker vs FixedSizeChunker (baseline)
# ============================================================
def run_recursive_vs_baseline(docs):
    print("\n" + "=" * 70)
    print("PHẦN 3: SO SÁNH RecursiveChunker (Strategy 3) vs FixedSizeChunker (Baseline)")
    print("=" * 70)

    # Baseline: FixedSizeChunker(chunk_size=500, overlap=50)
    baseline = FixedSizeChunker(chunk_size=500, overlap=50)
    # Strategy 3: RecursiveChunker(chunk_size=500) — separators mặc định
    strategy3 = RecursiveChunker(chunk_size=500)

    for doc in docs:
        baseline_chunks = baseline.chunk(doc.content)
        recursive_chunks = strategy3.chunk(doc.content)

        b_avg = sum(len(c) for c in baseline_chunks) / len(baseline_chunks) if baseline_chunks else 0
        r_avg = sum(len(c) for c in recursive_chunks) / len(recursive_chunks) if recursive_chunks else 0

        print(f"\n--- {doc.metadata['title']} ({len(doc.content)} chars) ---")
        print(f"  Baseline (FixedSize):   count={len(baseline_chunks):3d}, avg={b_avg:.0f}")
        print(f"  Strategy 3 (Recursive): count={len(recursive_chunks):3d}, avg={r_avg:.0f}")

    # Sample chunks comparison
    sample_doc = docs[2] if len(docs) > 2 else docs[0]  # chinhsachthuepin
    print(f"\n--- SAMPLE CHUNKS: {sample_doc.metadata['title']} ---")

    baseline_chunks = baseline.chunk(sample_doc.content)
    recursive_chunks = strategy3.chunk(sample_doc.content)

    print(f"\n  [Baseline] Top 3 chunks:")
    for i, c in enumerate(baseline_chunks[:3]):
        preview = c[:120].replace('\n', '\\n')
        print(f"    Chunk {i+1} ({len(c)} chars): {preview}...")

    print(f"\n  [RecursiveChunker] Top 3 chunks:")
    for i, c in enumerate(recursive_chunks[:3]):
        preview = c[:120].replace('\n', '\\n')
        print(f"    Chunk {i+1} ({len(c)} chars): {preview}...")


# ============================================================
# 5. Benchmark Queries — Chạy 5 queries với RecursiveChunker
# ============================================================
BENCHMARK_QUERIES = [
    {
        "query": "Quyền và nghĩa vụ của khách hàng khi thuê pin VinFast là gì?",
        "gold_answer": "Khách hàng có quyền được sửa chữa, thay thế Pin khi SOH dưới 70%. Có nghĩa vụ thanh toán đúng hạn và tuân thủ điều kiện sử dụng và bảo quản Pin.",
        "expected_category": "battery_rental",
    },
    {
        "query": "VinFast thu thập những loại dữ liệu cá nhân nào?",
        "gold_answer": "VinFast thu thập dữ liệu cá nhân cơ bản (họ tên, ngày sinh, CCCD, địa chỉ, email, SĐT) và dữ liệu cá nhân nhạy cảm (quan điểm chính trị, sức khỏe, tài khoản ngân hàng, dữ liệu vị trí).",
        "expected_category": "privacy",
    },
    {
        "query": "Điều kiện để đặt cọc mua xe VinFast trên website là gì?",
        "gold_answer": "Cá nhân từ đủ 18 tuổi trở lên có năng lực hành vi dân sự, hoặc tổ chức được thành lập hợp pháp theo pháp luật Việt Nam.",
        "expected_category": "deposit",
    },
    {
        "query": "Chính sách hoàn tiền đặt cọc xe VinFast trong trường hợp nào?",
        "gold_answer": "VinFast hoàn tiền đặt cọc trong trường hợp VinFast không giao xe đúng hạn hoặc hủy đơn hàng. Nếu khách hàng tự hủy, tiền cọc không được hoàn lại.",
        "expected_category": "deposit",
    },
    {
        "query": "Phí dịch vụ thuê pin ô tô điện VinFast được tính như thế nào?",
        "gold_answer": "Phí dịch vụ cho thuê pin được tính dựa trên quãng đường di chuyển thực tế (km) hoặc phí cố định hàng tháng, tùy gói thuê pin mà khách hàng chọn.",
        "expected_category": "battery_rental",
    },
]


def run_benchmark_queries(docs):
    print("\n" + "=" * 70)
    print("PHẦN 4: BENCHMARK QUERIES — RecursiveChunker (Strategy 3)")
    print("=" * 70)

    # --- BASELINE store: FixedSizeChunker ---
    baseline_chunker = FixedSizeChunker(chunk_size=500, overlap=50)
    baseline_store = EmbeddingStore(collection_name="baseline_store", embedding_fn=_mock_embed)

    for doc in docs:
        chunks = baseline_chunker.chunk(doc.content)
        chunk_docs = [
            Document(
                id=f"{doc.id}_chunk{i}",
                content=chunk,
                metadata={**doc.metadata, "chunk_index": i},
            )
            for i, chunk in enumerate(chunks)
        ]
        baseline_store.add_documents(chunk_docs)

    # --- STRATEGY 3 store: RecursiveChunker ---
    recursive_chunker = RecursiveChunker(chunk_size=500)
    strategy3_store = EmbeddingStore(collection_name="strategy3_store", embedding_fn=_mock_embed)

    for doc in docs:
        chunks = recursive_chunker.chunk(doc.content)
        chunk_docs = [
            Document(
                id=f"{doc.id}_chunk{i}",
                content=chunk,
                metadata={**doc.metadata, "chunk_index": i},
            )
            for i, chunk in enumerate(chunks)
        ]
        strategy3_store.add_documents(chunk_docs)

    print(f"\n  Baseline store size: {baseline_store.get_collection_size()} chunks")
    print(f"  Strategy3 store size: {strategy3_store.get_collection_size()} chunks")

    # Demo LLM
    def demo_llm(prompt):
        return f"[DEMO LLM] Trả lời dựa trên context đã retrieve."

    baseline_agent = KnowledgeBaseAgent(store=baseline_store, llm_fn=demo_llm)
    strategy3_agent = KnowledgeBaseAgent(store=strategy3_store, llm_fn=demo_llm)

    for i, bq in enumerate(BENCHMARK_QUERIES, 1):
        query = bq["query"]
        gold = bq["gold_answer"]
        expected_cat = bq["expected_category"]

        print(f"\n{'='*60}")
        print(f"  Query {i}: {query}")
        print(f"  Gold Answer: {gold[:100]}...")
        print(f"  Expected Category: {expected_cat}")

        # --- Baseline results ---
        baseline_results = baseline_store.search(query, top_k=3)
        print(f"\n  [BASELINE — FixedSizeChunker]")
        for j, r in enumerate(baseline_results, 1):
            cat = r['metadata'].get('category', '?')
            preview = r['content'][:100].replace('\n', '\\n')
            relevant = "✅" if cat == expected_cat else "❌"
            print(f"    Top-{j}: score={r['score']:.4f} | cat={cat} {relevant} | {preview}...")

        # --- Strategy 3 results ---
        strategy3_results = strategy3_store.search(query, top_k=3)
        print(f"\n  [STRATEGY 3 — RecursiveChunker]")
        for j, r in enumerate(strategy3_results, 1):
            cat = r['metadata'].get('category', '?')
            preview = r['content'][:100].replace('\n', '\\n')
            relevant = "✅" if cat == expected_cat else "❌"
            print(f"    Top-{j}: score={r['score']:.4f} | cat={cat} {relevant} | {preview}...")

        # --- With metadata filter ---
        filtered_results = strategy3_store.search_with_filter(
            query, top_k=3, metadata_filter={"category": expected_cat}
        )
        print(f"\n  [STRATEGY 3 + FILTER category={expected_cat}]")
        for j, r in enumerate(filtered_results, 1):
            cat = r['metadata'].get('category', '?')
            preview = r['content'][:100].replace('\n', '\\n')
            print(f"    Top-{j}: score={r['score']:.4f} | cat={cat} | {preview}...")


# ============================================================
# 6. Chunk coherence analysis — kiểm tra tính trọn vẹn ý
# ============================================================
def run_coherence_analysis(docs):
    print("\n" + "=" * 70)
    print("PHẦN 5: CHUNK COHERENCE ANALYSIS — RecursiveChunker vs FixedSizeChunker")
    print("=" * 70)

    # Chọn tài liệu điều khoản đặt cọc (có cấu trúc rõ)
    target_doc = None
    for doc in docs:
        if doc.id == "dieukhoandatcoc":
            target_doc = doc
            break

    if target_doc is None:
        print("  [SKIP] dieukhoandatcoc.md not found")
        return

    baseline = FixedSizeChunker(chunk_size=500, overlap=50)
    strategy3 = RecursiveChunker(chunk_size=500)

    b_chunks = baseline.chunk(target_doc.content)
    r_chunks = strategy3.chunk(target_doc.content)

    print(f"\n  Document: {target_doc.metadata['title']} ({len(target_doc.content)} chars)")
    print(f"  Baseline chunks: {len(b_chunks)}")
    print(f"  Recursive chunks: {len(r_chunks)}")

    # Count chunks that start/end mid-sentence (heuristic: check if ends with punctuation)
    def coherence_score(chunks):
        good = 0
        for c in chunks:
            stripped = c.strip()
            if stripped and stripped[-1] in '.。:：\n':
                good += 1
        return good / len(chunks) * 100 if chunks else 0

    b_score = coherence_score(b_chunks)
    r_score = coherence_score(r_chunks)

    print(f"\n  Coherence Score (% chunks ending at sentence boundary):")
    print(f"    Baseline:  {b_score:.1f}%")
    print(f"    Recursive: {r_score:.1f}%")

    print(f"\n  Phân tích: RecursiveChunker tách theo paragraph (\\n\\n) trước,")
    print(f"  rồi mới tách theo dòng (\\n), rồi câu (\". \"),")
    print(f"  → chunk có xu hướng giữ nguyên ý của đoạn văn bản.")
    print(f"  FixedSizeChunker cắt cứng tại vị trí ký tự → có thể cắt giữa câu.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("🚀 Strategy 3 — RecursiveChunker Benchmark")
    print(f"   Separators: \\n\\n → \\n → \". \" → \" \" → \"\"")
    print()

    docs = load_vinfast_docs()
    print(f"Loaded {len(docs)} documents:")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['title']} ({len(doc.content)} chars)")

    run_strategy_comparison(docs)
    run_parameter_tuning(docs)
    run_recursive_vs_baseline(docs)
    run_benchmark_queries(docs)
    run_coherence_analysis(docs)

    print("\n" + "=" * 70)
    print("✅ Benchmark hoàn tất!")
    print("=" * 70)

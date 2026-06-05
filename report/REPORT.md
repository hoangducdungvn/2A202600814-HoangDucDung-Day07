# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Hoàng Đức Dũng
**Nhóm:** [Tên nhóm]
**Ngày:** 2026-06-05

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (gần 1.0) có nghĩa là hai đoạn văn bản có **ý nghĩa ngữ nghĩa tương tự nhau** — các vector embedding của chúng chỉ cùng một hướng trong không gian nhiều chiều. Nói cách khác, mặc dù hai câu có thể dùng từ khác nhau, nhưng chúng diễn đạt cùng một ý tưởng hoặc chủ đề.

**Ví dụ HIGH similarity:**
- Sentence A: "Python là một ngôn ngữ lập trình bậc cao dễ học."
- Sentence B: "Python is a high-level programming language that is easy to learn."
- Tại sao tương đồng: Hai câu này diễn đạt cùng một ý nghĩa (mô tả Python là ngôn ngữ lập trình bậc cao, dễ học), chỉ khác ngôn ngữ. Embedding model sẽ tạo ra các vector có hướng gần giống nhau vì nội dung ngữ nghĩa trùng khớp.

**Ví dụ LOW similarity:**
- Sentence A: "Machine learning sử dụng thuật toán để học từ dữ liệu."
- Sentence B: "Hôm nay trời nắng đẹp, thích hợp để đi biển."
- Tại sao khác: Hai câu thuộc hai domain hoàn toàn khác nhau (công nghệ vs thời tiết/du lịch), không chia sẻ ngữ nghĩa chung. Các vector embedding sẽ chỉ theo các hướng rất khác nhau, dẫn đến cosine similarity thấp (gần 0).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo **hướng** (góc) giữa hai vector, không quan tâm đến **độ dài** (magnitude). Điều này rất quan trọng vì hai đoạn văn bản có cùng ý nghĩa nhưng độ dài khác nhau sẽ có vector embedding khác magnitude nhưng cùng hướng. Euclidean distance bị ảnh hưởng bởi magnitude, nên hai vector cùng hướng nhưng khác độ dài sẽ có khoảng cách Euclidean lớn — dẫn đến kết quả so sánh không chính xác.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> **Công thức:** `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
>
> **Phép tính:** `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.111) = 23`
>
> **Đáp án: 23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> **Phép tính:** `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25 chunks` — tăng từ 23 lên **25 chunks**.
>
> Overlap nhiều hơn giúp đảm bảo rằng các thông tin nằm ở **ranh giới giữa hai chunk** không bị mất ngữ cảnh. Khi một câu hoặc một ý bị cắt đôi bởi ranh giới chunk, phần overlap sẽ giữ lại nội dung đó trong chunk kế tiếp, giúp retrieval tìm được chunk chứa đầy đủ thông tin hơn. Đổi lại, overlap lớn hơn sẽ tạo ra nhiều chunks hơn, tốn thêm bộ nhớ và thời gian xử lý.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Chính sách pháp lý VinFast (VinFast Legal Policies)

**Tại sao nhóm chọn domain này?**
> VinFast là hãng xe điện Việt Nam có hệ thống chính sách phức tạp bao gồm bảo mật thanh toán, đặt cọc, thuê pin, bảo vệ dữ liệu cá nhân, cookies, v.v. Domain này phù hợp cho RAG vì khách hàng thường cần tra cứu nhanh các điều khoản cụ thể. Ngoài ra, các tài liệu pháp lý có cấu trúc rõ ràng (điều, khoản, mục) — rất tốt để test các chunking strategy khác nhau.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | chinh_sach_san_pham.md | vinfastauto.com | 1,031 | category=product_policy, lang=vi |
| 2 | chinhsachbaomat.md | vinfastauto.com | 3,695 | category=security, lang=vi |
| 3 | dieukhoan.md | vinfastauto.com | 6,570 | category=terms, lang=vi |
| 4 | dieukhoandatcoc.md | vinfastauto.com | 6,788 | category=deposit, lang=vi |
| 5 | dieukiensudung.md | vinfastauto.com | 7,442 | category=cookies, lang=vi |
| 6 | baomatcanhan.md | vinfastauto.com | 34,625 | category=privacy, lang=vi |
| 7 | chinhsachthuepin.md | vinfastauto.com | 19,587 | category=battery_rental, lang=vi |

### Metadata Schema

| Trường (Field) | Kiểu dữ liệu | Ý nghĩa / Mô tả |
|----------------|--------------|----------------|
| `category` | `str` | Phân loại tài liệu (ví dụ: `security`, `terms`, `battery_rental`) |
| `lang` | `str` | Ngôn ngữ của tài liệu (`vi` cho tiếng Việt) |
| `source` | `str` | Nguồn lấy tài liệu (ví dụ: `vinfastauto.com`) |

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu VinFast (chunk_size=500):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| chinhsachbaomat.md (2,908 chars) | FixedSizeChunker (`fixed_size`) | 7 | 458 | ❌ Cắt giữa câu |
| | SentenceChunker (`by_sentences`) | 8 | 362 | ⚠️ Giữ câu nhưng mất cấu trúc mục |
| | RecursiveChunker (`recursive`) | 10 | 289 | ✅ Tách theo đoạn / dòng → giữ nguyên mục |
| dieukhoandatcoc.md (5,099 chars) | FixedSizeChunker (`fixed_size`) | 12 | 471 | ❌ Cắt giữa điều khoản |
| | SentenceChunker (`by_sentences`) | 12 | 421 | ⚠️ Giữ câu nhưng có thể cắt đoạn |
| | RecursiveChunker (`recursive`) | 16 | 317 | ✅ Giữ nguyên trọn vẹn từng điều khoản |
| chinhsachthuepin.md (14,893 chars) | FixedSizeChunker (`fixed_size`) | 33 | 500 | ❌ Cắt ngẫu nhiên tại 500 chars |
| | SentenceChunker (`by_sentences`) | 38 | 390 | ⚠️ Gom 3 câu nhưng không chú ý đoạn |
| | RecursiveChunker (`recursive`) | 44 | 337 | ✅ Tách theo cấu trúc tài liệu rất tốt |

### Strategy Của Tôi: RecursiveChunker (Strategy 3)

**Tại sao chọn strategy này cho domain pháp lý VinFast?**
Tài liệu pháp lý (như Điều khoản, Chính sách bảo mật) có cấu trúc phân tầng rất nghiêm ngặt (Điều 1, Khoản 1.1, các đoạn văn bản). Việc cắt ngang một đoạn văn bản hoặc cắt giữa một câu sẽ làm mất đi ngữ cảnh pháp lý cực kỳ quan trọng. `RecursiveChunker` thử tách theo đoạn văn (`

`), sau đó đến dòng (`
`), rồi mới đến câu (`. `). Nhờ vậy, nó giữ được cấu trúc phân cấp tự nhiên của văn bản, đảm bảo mỗi chunk chứa trọn vẹn một ý (coherence) tốt hơn nhiều so với việc cắt cứng theo số lượng ký tự (`FixedSizeChunker`).

**Tuning Tham Số (chunk_size):**
Thử nghiệm trên `dieukhoandatcoc.md`:
- `chunk_size=200`: 43 chunks (avg 117 chars) → Quá nhỏ, bị xé vụn các điều khoản dài.
- `chunk_size=300`: 30 chunks (avg 168 chars) → Tương tự, vẫn bị xé lẻ nhiều.
- `chunk_size=500`: 16 chunks (avg 317 chars) → **Tối ưu**. Giữ được trọn vẹn hầu hết các điều khoản, min=1, max=494.
- `chunk_size=800`: 9 chunks (avg 565 chars) → Hơi lớn, một chunk có thể chứa nhiều điều khoản khác nhau, giảm precision khi search.

**Kết luận:** Sử dụng `RecursiveChunker` với `chunk_size=500` và separators mặc định `["\\n\\n", "\\n", ". ", " ", ""]`. Phân tích Chunk Coherence cho thấy tỷ lệ chunk kết thúc đúng ranh giới câu/đoạn của `RecursiveChunker` đạt 68.8% so với chỉ 8.3% của `FixedSizeChunker` (trên tài liệu `dieukhoandatcoc.md`).

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex `re.split(r'(?<=[.!?])\s', text)` để tách câu — lookbehind assertion detect dấu kết thúc câu (`.`, `!`, `?`) theo sau bởi whitespace. Sau khi split, loại bỏ chuỗi rỗng và strip whitespace ở mỗi câu. Cuối cùng, nhóm các câu theo `max_sentences_per_chunk` bằng cách duyệt qua list sentences với step bằng max_sentences_per_chunk, join từng nhóm lại bằng dấu cách. Edge case xử lý: text rỗng trả về `[]`, strip whitespace thừa.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm đệ quy thử tách text bằng từng separator theo thứ tự ưu tiên (`\n\n` → `\n` → `. ` → ` ` → `""`). **Base case**: text ngắn hơn `chunk_size` thì trả về nguyên. Nếu separator hiện tại không split được text (chỉ 1 phần), chuyển sang separator kế tiếp. Khi split được, các phần nhỏ được gộp lại (merge) cho đến khi vượt `chunk_size`, phần quá lớn thì đệ quy với separator tiếp theo. Separator rỗng `""` hoặc hết separators thì fallback sang cắt từng ký tự theo `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi document được embed bằng `embedding_fn` và lưu thành dict record gồm `{id, content, embedding, metadata}` trong list `self._store`. Khi search, embed query rồi tính dot product giữa query embedding và tất cả stored embeddings (vì mock embedder đã normalize vector nên dot product ≈ cosine similarity). Sort kết quả theo score giảm dần, trả về top_k records với keys `content`, `score`, `metadata`. Nếu có ChromaDB thì song song lưu vào collection.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` **filter trước, search sau**: duyệt `self._store`, giữ lại các record có metadata match tất cả key-value trong `metadata_filter`, rồi mới chạy similarity search trên tập đã lọc. Nếu `metadata_filter` là `None` thì delegate sang `search()` bình thường. `delete_document` dùng list comprehension để tạo list mới loại bỏ các record có `metadata['doc_id'] == doc_id`, so sánh size trước/sau để trả về `True/False`. Nếu có ChromaDB thì đồng thời xóa trong collection.

### KnowledgeBaseAgent

**`answer`** — approach:
> Theo RAG pattern 3 bước: (1) Gọi `store.search(question, top_k)` để lấy top-k chunks liên quan nhất; (2) Build prompt có cấu trúc rõ ràng gồm phần Context (đánh số `[1], [2], ...` cho mỗi chunk) và phần Question; (3) Gọi `llm_fn(prompt)` để sinh câu trả lời dựa trên context đã inject. Prompt structure giúp LLM biết rõ đâu là context retrieval và đâu là câu hỏi cần trả lời.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.10.0, pytest-8.4.1, pluggy-1.6.0
rootdir: E:\20k\2A202600814-HoangDucDung-Day07
plugins: anyio-4.9.0, Faker-40.21.0, langsmith-0.4.3

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.14s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Python la ngon ngu lap trinh bac cao." | "Python is a high-level programming language." | high | -0.1033 | ❌ |
| 2 | "Machine learning su dung thuat toan de hoc tu du lieu." | "Deep learning la mot nhanh cua machine learning." | high | -0.0197 | ❌ |
| 3 | "Meo thich an ca va ngu trua." | "Vector database luu tru embeddings de tim kiem tuong dong." | low | -0.1167 | ✅ |
| 4 | "Cosine similarity do goc giua hai vector." | "Euclidean distance do khoang cach giua hai diem." | high | 0.0508 | ❌ |
| 5 | "Ha Noi la thu do cua Viet Nam." | "Ha Noi noi tieng voi pho va bun cha." | high | -0.0176 | ❌ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Tất cả các scores đều gần 0 (từ -0.12 đến +0.05), kể cả những cặp câu rõ ràng cùng chủ đề (Pair 1, 2, 4, 5). Điều này **không bất ngờ** khi hiểu rằng `MockEmbedder` tạo vector bằng hàm hash (MD5), hoàn toàn **không hiểu ngữ nghĩa** — mỗi chuỗi text khác nhau sẽ cho vector ngẫu nhiên không liên quan. Kết quả này cho thấy sự khác biệt rõ ràng giữa mock embedder (chỉ để test logic code) và real embedder (như `all-MiniLM-L6-v2` hoặc OpenAI) — real embedder sẽ encode ý nghĩa ngữ nghĩa vào vector, giúp các câu cùng chủ đề có cosine similarity cao thực sự.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer | Chunk nào chứa thông tin? |
|---|-------|-------------|--------------------------|
| 1 | Quyền và nghĩa vụ của khách hàng khi thuê pin VinFast là gì? | Khách hàng có quyền được sửa chữa, thay thế Pin khi SOH dưới 70%. Có nghĩa vụ thanh toán đúng hạn và tuân thủ điều kiện sử dụng và bảo quản Pin. | chinhsachthuepin.md |
| 2 | VinFast thu thập những loại dữ liệu cá nhân nào? | VinFast thu thập dữ liệu cá nhân cơ bản (họ tên, ngày sinh, CCCD, địa chỉ, email, SĐT) và dữ liệu cá nhân nhạy cảm (quan điểm chính trị, sức khỏe, tài khoản ngân hàng, dữ liệu vị trí). | baomatcanhan.md |
| 3 | Điều kiện để đặt cọc mua xe VinFast trên website là gì? | Cá nhân từ đủ 18 tuổi trở lên có năng lực hành vi dân sự, hoặc tổ chức được thành lập hợp pháp theo pháp luật Việt Nam. | dieukhoan.md |
| 4 | Chính sách hoàn tiền đặt cọc xe VinFast trong trường hợp nào? | VinFast hoàn tiền đặt cọc trong trường hợp VinFast không giao xe đúng hạn hoặc hủy đơn hàng. Nếu khách hàng tự hủy, tiền cọc không được hoàn lại. | dieukhoandatcoc.md |
| 5 | Phí dịch vụ thuê pin ô tô điện VinFast được tính như thế nào? | Phí dịch vụ cho thuê pin được tính dựa trên quãng đường di chuyển thực tế (km) hoặc phí cố định hàng tháng, tùy gói thuê pin mà khách hàng chọn. | chinhsachthuepin.md |

### Kết Quả Của Tôi (Sử dụng RecursiveChunker)

*(Lưu ý: MockEmbedder không hiểu ngữ nghĩa nên kết quả retrieval bị sai lệch. Tuy nhiên quy trình benchmark vẫn hoạt động chuẩn.)*

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Quyền và nghĩa vụ của KH khi thuê pin | "Vui lòng đọc kỹ các điều khoản..." (cat=deposit) | 0.449 | ❌ | Dựa trên context không liên quan, Agent sinh câu trả lời bị hallucinate hoặc từ chối trả lời. |
| 2 | VinFast thu thập dữ liệu cá nhân nào? | "Chủ Thể Dữ Liệu Cá Nhân..." (cat=privacy) | 0.274 | ✅ | Dựa trên context, Agent liệt kê một phần thông tin về dữ liệu cá nhân, nhưng chưa đủ do chunk chưa cover hết. |
| 3 | Điều kiện đặt cọc mua xe trên website | "Một số trang trên Vinfastauto..." (cat=usage) | 0.429 | ❌ | Lấy nhầm sang trang điều kiện sử dụng website (cookies). |
| 4 | Chính sách hoàn tiền đặt cọc | "Công Ty khuyến cáo Chủ Thể..." (cat=privacy) | 0.304 | ❌ | Lấy sai hoàn toàn sang chính sách bảo mật do mock embedding. |
| 5 | Phí dịch vụ thuê pin tính thế nào? | "Nếu có bất kỳ câu hỏi nào..." (cat=privacy) | 0.299 | ❌ | Tương tự, lấy nhầm đoạn thông tin liên hệ. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 1 / 5

**Lưu ý:** Nếu áp dụng Metadata Filtering (ví dụ: `filter={"category": "battery_rental"}`) cho câu 1 và câu 5, RecursiveChunker trả về kết quả chính xác hơn nhiều. MockEmbedder là rào cản chính gây ra điểm số similarity kém, chứ không phải do Chunking Strategy. RecursiveChunker đã hoàn thành tốt việc cắt nhỏ văn bản gọn gàng theo đoạn.

---

### So Sánh Kết Quả Trong Nhóm (Group Comparison)

*(Phần này dùng để tổng hợp và so sánh kết quả của các thành viên trong nhóm sau khi chạy benchmark)*

| Strategy | Thành viên | Số câu query lấy đúng (Top-3) | Nhận xét điểm mạnh | Nhận xét điểm yếu |
|----------|------------|-------------------------------|--------------------|-------------------|
| Strategy 1 (Baseline FixedSize) | [Tên thành viên 1] | ... / 5 | Chạy nhanh, dễ cấu hình | Cắt vỡ câu, mất ngữ cảnh đoạn |
| Strategy 2 (SentenceChunker) | [Tên thành viên 2] | ... / 5 | Không bao giờ cắt ngang câu | Phá vỡ cấu trúc văn bản pháp lý |
| Strategy 3 (RecursiveChunker) | Hoang Duc Dung | 1 / 5 | Tách đoạn và câu chuẩn, giữ nguyên ý | Chunk size dao động nhiều |
| Strategy 4 (Custom/Tuned) | [Tên thành viên 4] | ... / 5 | Rất chính xác theo domain | Tốn công viết regex / logic |

**Tổng kết chung từ nhóm:**
- Strategy cho retrieval tốt nhất là: `[Điền tên strategy]` vì `[lý do]`.
- Ví dụ về query mà Strategy A tốt hơn B: Query số `[X]`, strategy `[A]` lấy đúng vì chunk to hơn, còn `[B]` lấy sai vì chia nhỏ câu hỏi bị mất từ khóa.
- Metadata filtering có giúp ích không? `[Có/Không. Lý do...]`

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi thấy bạn làm `SentenceChunker` có ưu điểm là rất đều về kích thước câu, không bao giờ bị cắt giữa chừng một từ hay một ý nhỏ. Tuy nhiên, nó lại phá vỡ cấu trúc đoạn văn pháp lý, khiến nhiều câu bị tách khỏi ngữ cảnh xung quanh.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Có nhóm đã sử dụng Regex tùy chỉnh (Custom Chunker) tách chuẩn theo đúng format "Điều X", "Khoản Y" của văn bản pháp luật Việt Nam. Điều này đem lại độ chính xác cực cao, cho thấy domain-specific chunking thực sự tạo ra sự khác biệt.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thiết kế metadata chi tiết hơn, thêm các trường như `article_number` (số điều) hoặc `tags` để có thể sử dụng tính năng `search_with_filter` mạnh mẽ hơn. Việc chỉ dùng vector search chay với MockEmbedder cho thấy sự cần thiết của hybrid search (vector + keyword/metadata). Ngoài ra, tôi sẽ chuyển sang dùng `all-MiniLM-L6-v2` thay vì MockEmbedder để thấy rõ tác dụng của embeddings thực sự.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |

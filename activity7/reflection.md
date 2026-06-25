# Reflection Answers

## 1. Which chunking strategy returned the most relevant text?

The Fixed-Size Chunking strategy returned the most relevant text for the query. However, the sentence was cut off because the chunk reached its character limit. Some context was lost in the result.

---

## 2. What happened to the text structure in Fixed-Size Chunk #2 vs. Paragraph Chunk #2?

Fixed-Size Chunk #2 split the text in the middle of sentences because it follows a fixed number of characters. Paragraph Chunk #2 kept the whole paragraph together, making it easier to understand and preserving the complete meaning.

---

## 3. Why is relying only on Fixed-Size Chunking risky for an HR handbook?

Important policies or instructions may be split between chunks. This can cause the AI to retrieve incomplete information and give inaccurate answers to employees.

---

## 4. Why store metadata such as `chunk_index` and `strategy`?

Metadata helps identify where a chunk came from and which chunking method was used. It also makes debugging easier and helps organize the retrieved information.
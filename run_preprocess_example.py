"""Run preprocess_corpus on repo data and print example corpus with Chinese tokens."""

from preprocess_corpus import DEFAULT_CONFIG, format_bow_doc, preprocess_corpus

# Override ingestion for local repo (Scrape path used in production config)
config = {
    **DEFAULT_CONFIG,
    "data_dir": "data",
    "file_pattern": "comments_oid323836485.csv",
    "vote_values": None,
}

result = preprocess_corpus(config)
dictionary = result["dictionary"]
corpus = result["corpus"]
lda_docs = result["lda_docs"]
stats = result["stats"]

print("=" * 60)
print("Validation stats")
print("=" * 60)
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("Sample vocabulary (first 30 tokens, Chinese)")
print("=" * 60)
for i, token in enumerate(list(dictionary.values())[:30]):
    print(f"  {token}", end="  ")
print()

print("\n" + "=" * 60)
print("Example document 1 — token list (Chinese)")
print("=" * 60)
if lda_docs:
    print(lda_docs[0][:25])

print("\n" + "=" * 60)
print("Example document 1 — BoW with Chinese tokens")
print("=" * 60)
if corpus:
    for token, count in format_bow_doc(corpus[0], dictionary):
        print(f"  {token}: {count}")

print("\n" + "=" * 60)
print("Example document 2 — BoW with Chinese tokens")
print("=" * 60)
if len(corpus) > 1:
    for token, count in format_bow_doc(corpus[1], dictionary):
        print(f"  {token}: {count}")

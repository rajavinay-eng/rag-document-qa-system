import chromadb
import time

# Persistent database
client = chromadb.PersistentClient(path="./project2_db")

# Create collection
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

print("ChromaDB setup complete")
print("Project 2 database ready")

# Sample chunks
chunks = [
    "The medication showed 94% efficacy in Phase 3 clinical trials.",
    "Common side effects include nausea in 15% of patients.",
    "Patients with kidney disease require dosage adjustment.",
    "Store medication between 2 and 8 degrees Celsius.",
    "FDA approved the medication in March 2024."
]

# Metadata
metadatas = [
    {"doc_id": "clinical_trial", "chunk": i}
    for i in range(len(chunks))
]

# IDs
ids = [f"chunk_{i}" for i in range(len(chunks))]

# Remove old chunks if already present
try:
    collection.delete(ids=ids)
except:
    pass

# Add to database WITHOUT embeddings
collection.add(
    documents=chunks,
    metadatas=metadatas,
    ids=ids
)

print(f"\nAdded {len(chunks)} chunks")

# SEARCH FUNCTION
def search_documents(query, n_results=2):

    start = time.time()

    # Search ChromaDB directly
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "distances"]
    )

    elapsed = (time.time() - start) * 1000

    print(f"\nQuery: {query}")
    print(f"\nSearch time: {elapsed:.1f} ms")

    print("\nTop Results:")

    for doc, dist in zip(
        results["documents"][0],
        results["distances"][0]
    ):
        similarity = 1 - dist

        print(f"\nSimilarity: {similarity:.3f}")
        print(doc)

# Test searches
search_documents("What are kidney risks?")
search_documents("How should medicine be stored?")
search_documents("What did the clinical trial show?")

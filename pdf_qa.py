
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to ChromaDB
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="pdf_qa"
)

# Load questions and chunks
queries = pd.read_csv("dataset/queries.csv")
chunks = pd.read_csv("dataset/chunks.csv")

# Connect to Ollama
llm = ChatOllama(model="llama3.2", temperature=0)

results = []

for _, row in queries.iterrows():
    question = row["question"]

    # Semantic search
    semantic_results = db.similarity_search_with_score(
        question, k=5
    )

    context = "\n\n".join(
        doc.page_content for doc, score in semantic_results
    )

    prompt = f"""
    Answer using ONLY the context below.
    If the answer is unavailable, say so.

    Context:
    {context}

    Question: {question}
    """

    answer = llm.invoke(prompt).content

    # Keyword-based search
    words = question.lower().split()
    keyword_scores = []

    for _, chunk in chunks.iterrows():
        text = str(chunk["text"])
        score = sum(text.lower().count(word) for word in words)
        keyword_scores.append((score, text))

    keyword_results = sorted(
        keyword_scores,
        key=lambda x: x[0],
        reverse=True
    )[:5]

    results.append({
        "query_id": row["query_id"],
        "question": question,
        "answer": answer,
        "semantic_top5": " | ".join(
            doc.page_content[:200]
            for doc, score in semantic_results
        ),
        "semantic_distances": " | ".join(
            str(round(float(score), 4))
            for doc, score in semantic_results
        ),
        "keyword_top5": " | ".join(
            text[:200] for score, text in keyword_results
        )
    })

    print("\nQUESTION:", question)
    print("\nANSWER:", answer)
    print("\nRETRIEVED CONTEXT:\n", context)

# Save results
pd.DataFrame(results).to_csv(
    "dataset/answers.csv", index=False
)

print("\nAll questions processed!")
print("Results saved in dataset/answers.csv")

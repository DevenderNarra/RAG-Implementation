import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------- LLM SETUP ---------------- #


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.5
)


# ---------------- RETRIEVER ---------------- #

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to vector DB...")
client = chromadb.PersistentClient(path="data/vectorstore")
collection = client.get_collection(name="pdf_documents")

print("\nRAG Chat Ready! (type exit to quit)\n")


# ---------------- CHAT LOOP ---------------- #

while True:

    query = input("Ask: ")

    if query.lower() == "exit":
        break

    # 1️⃣ Embed question
    query_embedding = model.encode([query], normalize_embeddings=True)

    # 2️⃣ Retrieve context
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )

    context = "\n\n".join(results["documents"][0])

    print("\n🔎 Retrieved Context Found")

    # 3️⃣ LLM Answer
    prompt = f"""
You are a professional assistant answering questions from a resume.

Use ONLY the context below to answer.

Context:
{context}

Question:
{query}

Rules:
- Be concise
- Do not hallucinate
- If not found say: Not mentioned in the resume
"""

    response = llm.invoke(prompt)

    print("\n🤖 Answer:\n")
    print(response.content)
    print("\n---------------------------------------------\n")

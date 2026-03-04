import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

class MemoryManager:
    def __init__(self, persist_directory="./db_faiss"):
        self.embeddings = OllamaEmbeddings(model="llama3.2", base_url="http://127.0.0.1:11434")
        self.persist_directory = persist_directory
        self.vector_store = None
        
        # Try to load existing local FAISS index
        if os.path.exists(os.path.join(self.persist_directory, "index.faiss")):
            try:
                self.vector_store = FAISS.load_local(
                    self.persist_directory, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Could not load local FAISS index: {e}")

    def add_memory(self, text: str, metadata: dict = None):
        """Adds a piece of information to the vector memory."""
        if metadata is None:
            metadata = {}
            
        if self.vector_store is None:
            # Lazy initialization
            self.vector_store = FAISS.from_texts([text], self.embeddings, metadatas=[metadata])
        else:
            self.vector_store.add_texts(texts=[text], metadatas=[metadata])
            
        # Persist to disk
        self.vector_store.save_local(self.persist_directory)

    def retrieve_memory(self, query: str, k: int = 3) -> list:
        """Retrieves relevant information given a query."""
        if self.vector_store is None:
            return []
            
        results = self.vector_store.similarity_search(query, k=k)
        return [{"content": res.page_content, "metadata": res.metadata} for res in results]

    def get_context(self, query: str, k: int = 2) -> str:
        """Helper to get and log semantic context."""
        memories = self.retrieve_memory(query, k=k)
        if not memories:
            return "No previous relevant context found."
            
        print(f"\n[Memory] Retrieved {len(memories)} relevant chunks for semantic grounding.")
        context = "\n---\n".join([m['content'] for m in memories])
        return context

# Singleton instance
memory = MemoryManager()

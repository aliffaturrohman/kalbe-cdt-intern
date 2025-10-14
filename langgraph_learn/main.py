from typing import TypedDict
from langgraph.graph import Graph
from langchain_ollama import ChatOllama

# 1. Definisikan State
class QAState(TypedDict):
    question: str
    answer: str

# 2. Inisialisasi model Ollama
model = ChatOllama(
    model="gemma3:1b-it-qat",
    temperature=0.7
)

# 3. Node untuk tanya-jawab
def ask_model(state: QAState) -> QAState:
    question = state.get("question", "")
    if not question:
        return {**state, "answer": "Tidak ada pertanyaan diberikan."}
    
    response = model.invoke(question)
    return {**state, "answer": response.content}

# 4. Bangun graph
graph = Graph()

# Tambah node
graph.add_node("ask_model", ask_model)

# Set entry point
graph.set_entry_point("ask_model")

# Karena hanya satu node, tidak perlu add_edge, tapi jika mau lanjut, bisa ditambahkan
# graph.add_edge("ask_model", "next_node")

# 5. Compile graph
app = graph.compile()

# 6. Contoh eksekusi
if __name__ == "__main__":
    result = app.invoke({"question": "Apa itu LangGraph?"})
    print(result["answer"])


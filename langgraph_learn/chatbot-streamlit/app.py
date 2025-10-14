import os
import re
import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient
from pymed_paperscraper import PubMed
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from typing import TypedDict

# # Load env
# load_dotenv()
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- State Definition ---
class State(TypedDict):
    query: str
    messages: list
    search_results: list
    sources: list
    keywords: list
    iteration: int

# --- LLM ---
llm = OllamaLLM(model="gemma3:1b-it-qat", temperature=0.5, top_k=3, top_p=5)
# llm = OllamaLLM(
#     model="gemma3:1b-it-qat",
#     temperature=0.5,
#     top_k=3,
#     top_p=5,
#     base_url=os.getenv("OLLAMA_HOST", "http://localhost:11435")
# )


# --- Prompts ---
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Your name is 'Yuzuriha'. You are will answer all question with indonesian language. "
                   "If you don't know about the answer, ."),
        ("human", "{chat_human}"),
    ]
)

classifier_internet_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Kamu adalah AI classifier. "
                   "Jika butuh data faktual/terkini/spesifik → 'NEED_SEARCH'. "
                   "Jika umum/konseptual → 'NO_SEARCH'. "
                   "Output hanya salah satu."),
        ("human", "{chat_human}"),
    ]
)

planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are AI planner. Output HARUS Python list inline. "
                   "Contoh: ['query1','query2']."),
        ("human", "{chat_human}"),
    ]
)

aggregator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Gabungkan hasil internet. Jawab singkat, jelas, bahasa Indonesia."),
        ("human", "Pertanyaan user: {query}\n\nHasil pencarian:\n{search_results}"),
    ]
)

pubmed_classifier_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Klasifikasi PubMed. "
                   "Jika pertanyaan tentang kesehatan → 'SEARCH_PUBMED'. "
                   "Selain itu → 'NO_PUBMED'."),
        ("human", "{chat_human}"),
    ]
)

pubmed_planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Ubah pertanyaan jadi query PubMed. "
                   "Output list Python inline."),
        ("human", "{chat_human}"),
    ]
)

pubmed_aggregator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Ringkas hasil abstrak PubMed. Bahasa Indonesia, singkat, "
                   "sebutkan keterbatasan bila ada."),
        ("human", "Pertanyaan user: {query}\n\nAbstract hasil PubMed:\n{abstracts}"),
    ]
)

# --- Graph Nodes ---
def chatbot(state: State):
    formatted_prompt = prompt.format_messages(chat_human=state["query"])
    response_text = "".join([t for t in llm.stream(formatted_prompt)])
    return {**state, "messages": state.get("messages", []) + [response_text]}

def llm_classifier(state: State):
    response = llm.invoke(classifier_internet_prompt.format_messages(chat_human=state["query"]))
    return {**state, "messages": state["messages"] + [response]}

def route_classifier(state: State):
    return "planner" if "NEED_SEARCH" in state["messages"][-1] else "model_llm"

def route_planner_loop(state: State):
    iteration = state.get("iteration", 0) + 1
    if len(state.get("search_results", [])) < 2 and iteration < 3:
        return "planner"
    return "aggregator_llm"

def planner_node(state: State):
    response = llm.invoke(planner_prompt.format_messages(chat_human=state["query"]))
    match = re.search(r"\[(.*?)\]", response)
    plan = [item.strip().strip("'\"") for item in match.group(1).split(",")] if match else []
    return {**state, "messages": state["messages"] + [f"Plan: {plan}"], "keywords": plan, "iteration": state.get("iteration", 0) + 1}

def executor_node(state: State):
    client = TavilyClient(api_key=TAVILY_API_KEY)
    results, sources = [], []
    for k in state.get("keywords", []):
        tool = client.search(query=k, max_results=3, topic="general")
        results.extend([r["content"] for r in tool["results"]])
        sources.extend([r["url"] for r in tool["results"]])
    return {**state, "search_results": results, "sources": sources, "messages": state["messages"] + [f"Found {len(results)} results"]}

def aggregator_llm(state: State):
    response = llm.invoke(aggregator_prompt.format_messages(query=state["query"], search_results="\n".join(state["search_results"])))
    return {**state, "messages": state["messages"] + [response]}

def pubmed_classifier(state: State):
    response = llm.invoke(pubmed_classifier_prompt.format_messages(chat_human=state["query"]))
    return {**state, "messages": state["messages"] + [response]}

def route_pubmed_classifier(state: State):
    return "pubmed_planner" if "SEARCH_PUBMED" in state["messages"][-1] else "internet_search_classifier"

def pubmed_planner_node(state: State):
    response = llm.invoke(pubmed_planner_prompt.format_messages(chat_human=state["query"]))
    match = re.search(r"\[(.*?)\]", response)
    plan = [item.strip().strip("'\"") for item in match.group(1).split(",")] if match else []
    return {**state, "keywords": plan, "messages": state["messages"] + [f"PubMed Plan: {plan}"]}

def pubmed_executor_node(state: State):
    pubmed = PubMed(tool="AlifScraper", email="aliffaturrohman11@gmail.com")
    abstracts, sources = [], []
    for k in state.get("keywords", []):
        results = pubmed.query(k, max_results=1)
        for article in results:
            if article.abstract:
                abstracts.append(article.abstract)
                sources.append(article.doi or article.journal)
    return {**state, "search_results": abstracts, "sources": sources, "messages": state["messages"] + [f"PubMed found {len(abstracts)} abstracts"]}

def pubmed_aggregator_llm(state: State):
    response = llm.invoke(pubmed_aggregator_prompt.format_messages(query=state["query"], abstracts="\n\n".join(state["search_results"])))
    return {**state, "messages": state["messages"] + [response]}

# --- Graph Definition ---
graph = StateGraph(State)
graph.set_entry_point("pubmed_classifier")
graph.add_node("pubmed_classifier", pubmed_classifier)
graph.add_node("pubmed_planner", pubmed_planner_node)
graph.add_node("pubmed_executor", pubmed_executor_node)
graph.add_node("pubmed_aggregator", pubmed_aggregator_llm)
graph.add_node("internet_search_classifier", llm_classifier)
graph.add_node("model_llm", chatbot)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("aggregator_llm", aggregator_llm)

graph.add_conditional_edges("pubmed_classifier", route_pubmed_classifier,
    {"pubmed_planner": "pubmed_planner", "internet_search_classifier": "internet_search_classifier"})
graph.add_edge("pubmed_planner", "pubmed_executor")
graph.add_edge("pubmed_executor", "pubmed_aggregator")
graph.add_edge("pubmed_aggregator", END)
graph.add_conditional_edges("internet_search_classifier", route_classifier,
    {"planner": "planner", "model_llm": "model_llm"})
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", route_planner_loop,
    {"planner": "planner", "aggregator_llm": "aggregator_llm"})
graph.add_edge("model_llm", END)
graph.add_edge("aggregator_llm", END)
app_graph = graph.compile()

# --- Streamlit UI ---
st.set_page_config(page_title="Yuzuriha Chatbot", layout="wide")
st.title("💬 Yuzuriha Chatbot")

if "history" not in st.session_state:
    st.session_state["history"] = []

query = st.chat_input("Tanyakan sesuatu...")
if query:
    state = {"query": query, "messages": [], "search_results": [], "sources": [], "iteration": 0}
    final_state = app_graph.invoke(state)
    answer = final_state["messages"][-1]

    st.session_state["history"].append((query, answer))
    st.session_state["history"] = st.session_state["history"][-4:]

for q, a in st.session_state["history"]:
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"): st.markdown(a)

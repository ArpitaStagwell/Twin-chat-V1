from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re, hashlib
from typing import Optional, List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from ollama import embeddings as ollama_embeddings, chat as ollama_chat
import chromadb
import logging

# -----------------------------
# Configs
# -----------------------------
PERSONA_PATH = "final_behavior_personas_ready_for_embedding.csv"
CHROMA_PATH = "chroma_persona_store"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest" #"mxbai-embed-large"
OLLAMA_CHAT_MODEL = "mistral"

CLUSTER_NAME_MAP = {
    0:"Affluent, Credit-active Digital Users",
    2:"High Net Worth Mail Responders",
    3:"High-Income Young Adult Credit Users",
    4:"Established Hispanic Consumers",
    5:"Young, Low-Credit, Tech-Savvy Consumers",
    6:"High-income Credit Active Professionals",
    7:"Moderate Credit Users with Frequent Fast Food Intake",
    8:"College-Educated Established Consumers",
    9:"Sub-Prime Credit Users",
    10:"Active Political Advocates & Community Involvers"
}

FEMALE_NAMES_POOL = ["Emma","Sophia","Olivia","Ava","Grace","Ella","Mia","Isabella","Amelia","Charlotte"]
MALE_NAMES_POOL = ["Liam","Noah","Ethan","James","Lucas","Henry","Leo","Mason","Logan","Benjamin"]
GENERIC_NAMES_POOL = FEMALE_NAMES_POOL + MALE_NAMES_POOL

# -----------------------------
# Load persona dataframe
# -----------------------------
persona_df = pd.read_csv(PERSONA_PATH).reset_index(drop=True)
persona_df["cluster_name"] = persona_df["behavior_cluster"].astype(int).map(CLUSTER_NAME_MAP)

def parse_age_range(a):
    if isinstance(a, str) and re.search(r"(\d{2})\s*(?:-|to)\s*(\d{2})", a):
        low, high = map(int, re.findall(r"(\d{2})", a)[:2])
        return low, high
    elif isinstance(a, (int, float)):
        return a, a
    return np.nan, np.nan

persona_df[["age_min","age_max"]] = persona_df["age_imputed"].apply(lambda x: pd.Series(parse_age_range(x)))

# -----------------------------
# ChromaDB (RAG) with startup check
# -----------------------------
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_collection("behavior_personas_store_v3")
    # Fetch first 3 entries as sanity check
    docs = chroma_collection.get()
    logging.info(f"Chroma collection loaded: {chroma_collection.name}")
    logging.info("First 3 metadata entries:")
    for meta in docs["metadatas"][:3]:
        logging.info(meta)
except Exception as e:
    logging.error(f"Error loading Chroma collection: {e}")
    raise RuntimeError(f"Cannot load Chroma collection: {e}")

# -----------------------------
# Ollama helpers
# -----------------------------
def ollama_summarize(prompt: str) -> str:
    res = ollama_chat(
        model=OLLAMA_CHAT_MODEL,
        messages=[{"role":"system","content":"You are a concise summarization engine."},
                  {"role":"user","content":prompt}],
        options={"temperature":0.4}
    )
    return res["message"]["content"].strip()

def deterministic_name_for_persona(persona_index: int, gender: Optional[str]) -> str:
    h = int(hashlib.sha1(str(persona_index).encode()).hexdigest()[:8], 16)
    if gender and str(gender).lower().startswith("m"):
        pool = MALE_NAMES_POOL
    elif gender and str(gender).lower().startswith("f"):
        pool = FEMALE_NAMES_POOL
    else:
        pool = GENERIC_NAMES_POOL
    return pool[h % len(pool)]

def smart_persona_brief(row: Dict[str, Any]) -> str:
    summary = str(row.get("persona_summary","")).strip()
    prompt = f"Summarize this persona in one line under 15 words:\n{summary}\nSummary:"
    compressed = ollama_summarize(prompt)
    return compressed

def get_cluster_llm_summary(cluster_id: int) -> str:
    cluster_data = persona_df[persona_df["behavior_cluster"]==cluster_id]["persona_summary"].str.cat(sep=" | ")
    prompt = f"Summarize this cluster in a single sentence under 20 words:\n{cluster_data}\nSummary:"
    return ollama_summarize(prompt)

def extract_gender_from_query(q: str) -> Optional[str]:
    q = q.lower()
    if re.search(r"\bmale|man|men\b", q): return "male"
    if re.search(r"\bfemale|woman|women\b", q): return "female"
    return None

def extract_age_range_from_query(q: str) -> Optional[tuple]:
    q = q.lower()
    match = re.search(r"(\d{2})\s*(?:-|to)\s*(\d{2})", q)
    if match: return int(match.group(1)), int(match.group(2))
    return None

def embed_texts_with_ollama(texts: List[str]) -> np.ndarray:
    return np.array([ollama_embeddings(model=OLLAMA_EMBED_MODEL, prompt=t)["embedding"] for t in texts])

# -----------------------------
# Core logic (no fallbacks)
# -----------------------------
def query_clusters_logic(query: str, top_k: int=3):
    clusters = persona_df.groupby(["behavior_cluster","cluster_name"])["persona_summary"].apply(" ".join).reset_index()
    cluster_embs = embed_texts_with_ollama(clusters["persona_summary"].tolist())
    q_emb = embed_texts_with_ollama([query])[0]
    sims = cosine_similarity([q_emb], cluster_embs)[0]
    clusters["similarity"] = sims
    return clusters.sort_values("similarity", ascending=False).head(top_k)

def filter_and_rank_personas_logic(cluster_personas: pd.DataFrame, query: str, top_k: int=5):
    df_copy = cluster_personas.copy()
    gender = extract_gender_from_query(query)
    age_range = extract_age_range_from_query(query)
    if gender: df_copy = df_copy[df_copy["gender_imputed"].str.lower().str.startswith(gender[0])]
    if age_range:
        low, high = age_range
        df_copy = df_copy[(df_copy["age_min"]<=high) & (df_copy["age_max"]>=low)]
    q_emb = embed_texts_with_ollama([query])[0]
    p_embs = embed_texts_with_ollama(df_copy["persona_summary"].tolist())
    sims = cosine_similarity([q_emb], p_embs)[0]
    df_copy["similarity"] = sims
    ranked = df_copy.sort_values("similarity", ascending=False).drop_duplicates(subset=["persona_summary"]).head(min(top_k,len(df_copy)))
    return ranked

def guardrail_filter(user_input: str, persona_name: str, persona_context: str, cluster_name: str) -> Optional[str]:
    # Only enforce bad-word and jailbreak detection (no greeting fallbacks)
    q = user_input.lower()
    bad_words = ["fuck","shit","ass","cunt","damn","hell","bitch","whore","pussy","dick","sex","nude","xxx","porn","wank","motherfucker","faggot","retard","asshole","idiot","stupid","nincompoop","moron","loser"]
    if any(b in q for b in bad_words):
        return f"{persona_name}: I cannot respond to that."
    return None

def persona_chat_logic(persona_id: int, user_message: str) -> str:
    persona = persona_df.loc[persona_id]
    persona_name = deterministic_name_for_persona(persona_id, persona.get("gender_imputed",""))
    cluster_name = persona.get("cluster_name","Unknown Cluster")
    persona_summary = persona.get("persona_summary","")

    # Guardrail
    guard = guardrail_filter(user_message, persona_name, persona_summary, cluster_name)
    if guard: return guard

    # RAG
    q_emb = embed_texts_with_ollama([f"{cluster_name} persona insights about: {user_message} and {persona_summary}"])[0].tolist()
    results = chroma_collection.query(query_embeddings=[q_emb], n_results=3)
    docs = results["documents"][0]
    docs = [d for d in docs if d != persona_summary]
    rag_context = "\n".join(docs)

    facts = f"""
You are {persona_name}, from cluster {cluster_name}.
Identity: {persona.get('gender_imputed','')} aged {persona.get('age_imputed','')}.
Details: {persona_summary}
Additional Context: {rag_context}
Answer concisely using ONLY the provided facts.
"""
    prompt = f"{facts}\nUser: {user_message}\n{persona_name}:"
    res = ollama_chat(model=OLLAMA_CHAT_MODEL, messages=[{"role":"user","content":prompt}])
    return res["message"]["content"]

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Persona Explorer API (Ollama, No Fallbacks)")

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class RefineRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "Persona Explorer API is running!"}

@app.get("/health")
def health_check():
    try:
        # Chroma test
        docs = chroma_collection.get(limit=1)
        # Ollama test (embedding call with a tiny prompt)
        emb = embed_texts_with_ollama(["test"])
        return {"status": "ok", "chroma_docs": len(docs["documents"]), "ollama_embedding_dim": len(emb[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_clusters")
def query_clusters(req: QueryRequest):
    df = query_clusters_logic(req.query, top_k=req.top_k)
    # convert numpy types to Python native
    result = df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")
    return result


@app.get("/cluster/{cluster_id}/personas")
def get_cluster_personas(cluster_id: int = Path(..., ge=0)):
    cluster_personas = persona_df[persona_df["behavior_cluster"]==cluster_id]
    return [{"persona_id":int(idx),
             "persona_name":deterministic_name_for_persona(idx,row.get("gender_imputed","")),
             "gender":row.get("gender_imputed",""),
             "age":row.get("age_imputed",""),
             "summary":row.get("persona_summary","")}
            for idx,row in cluster_personas.iterrows()]

@app.post("/cluster/{cluster_id}/refine_personas")
def refine_personas(cluster_id: int, req: RefineRequest):
    # Filter & rank
    df = filter_and_rank_personas_logic(persona_df[persona_df["behavior_cluster"]==cluster_id], req.query, top_k=req.top_k)
    
    # Convert to same output format as /personas
    result = [
        {
            "persona_id": int(idx),
            "persona_name": deterministic_name_for_persona(idx, row.get("gender_imputed","")),
            "gender": row.get("gender_imputed",""),
            "age": row.get("age_imputed",""),
            "summary": row.get("persona_summary","")
        }
        for idx, row in df.iterrows()
    ]
    return result


@app.get("/persona/{persona_id}/start_chat")
def start_persona_chat(persona_id: int):
    persona = persona_df.loc[persona_id]
    return {"greeting": f"I'm {deterministic_name_for_persona(persona_id, persona.get('gender_imputed',''))} from {persona.get('cluster_name','Unknown Cluster')}"}

@app.post("/persona/{persona_id}/chat")
def chat_with_persona(persona_id: int, req: ChatRequest):
    return {"reply": persona_chat_logic(persona_id, req.message)}

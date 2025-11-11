from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re, random, hashlib
from typing import Optional, List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

# Ollama & Chroma (RAG)
from ollama import embeddings as ollama_embeddings, chat as ollama_chat
import chromadb

# Try to optionally load a local summarizer (same approach as Streamlit)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# -----------------------------
# Configs (same names/paths as Streamlit)
# -----------------------------
PERSONA_PATH = "final_behavior_personas_ready_for_embedding.csv"
CHROMA_PATH = "chroma_persona_store"
OLLAMA_EMBED_MODEL = "mxbai-embed-large"
OLLAMA_CHAT_MODEL = "mistral"

CLUSTER_NAME_MAP = {
    0:"Affluent, Credit-active Digital Users"
    2:"High Net Worth Mail Responders"
    3:"High-Income Young Adult Credit Users"
    4:"Established Hispanic Consumers"
    5:"Young, Low-Credit, Tech-Savvy Consumers"
    6:"High-income Credit Active Professionals"
    7:"Moderate Credit Users with Frequent Fast Food Intake"
    8:"College-Educated Established Consumers"
    9:"Sub-Prime Credit Users"
    10:"Active Political Advocates & Community Involvers"
}

# Pools for deterministic persona names (used instead of Streamlit's session state)
FEMALE_NAMES_POOL = [
    "Emma", "Sophia", "Olivia", "Ava", "Grace",
    "Ella", "Mia", "Isabella", "Amelia", "Charlotte"
]
MALE_NAMES_POOL = [
    "Liam", "Noah", "Ethan", "James", "Lucas",
    "Henry", "Leo", "Mason", "Logan", "Benjamin"
]
GENERIC_NAMES_POOL = FEMALE_NAMES_POOL + MALE_NAMES_POOL

# -----------------------------
# Load persona dataframe and preprocess
# -----------------------------
try:
    persona_df = pd.read_csv(PERSONA_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Persona CSV not found at {PERSONA_PATH}")

# Ensure index is integer index matching how Streamlit used .loc[persona_id]
persona_df = persona_df.reset_index(drop=True)

# Map cluster names
persona_df["cluster_name"] = persona_df["behavior_cluster"].astype(int).map(CLUSTER_NAME_MAP)

# Age parsing utility (copied from streamlit)
def parse_age_range(a):
    if isinstance(a, str) and re.search(r"(\d{2})\s*(?:-|to)\s*(\d{2})", a):
        low, high = map(int, re.findall(r"(\d{2})", a)[:2])
        return low, high
    elif isinstance(a, (int, float)):
        return a, a
    return np.nan, np.nan

persona_df[["age_min", "age_max"]] = persona_df["age_imputed"].apply(
    lambda x: pd.Series(parse_age_range(x))
)

# -----------------------------
# ChromaDB (RAG) init (optional)
# -----------------------------
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_collection("behavior_personas_store_v3")
except Exception as e:
    chroma_collection = None

# -----------------------------
# Optional summarizer (try to follow streamlit behavior)
# -----------------------------
summarizer = None
llm_summarizer_available = False
if TRANSFORMERS_AVAILABLE:
    try:
        # same model name as Streamlit attempt (may fail if not available)
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.4)
        llm_summarizer_available = True
    except Exception:
        summarizer = None
        llm_summarizer_available = False

# -----------------------------
# Helper functions (preserve streamlit implementations)
# -----------------------------
def deterministic_name_for_persona(persona_index: int, gender: Optional[str]) -> str:
    """
    Deterministic mapping from persona index -> name (stateless).
    Tries to mimic assign_unique_name behavior but without session_state.
    """
    # use hash of index to pick name, but prefer gender-specific pool
    h = int(hashlib.sha1(str(persona_index).encode()).hexdigest()[:8], 16)
    if gender and str(gender).lower().startswith("m"):
        pool = MALE_NAMES_POOL
    elif gender and str(gender).lower().startswith("f"):
        pool = FEMALE_NAMES_POOL
    else:
        pool = GENERIC_NAMES_POOL
    name = pool[h % len(pool)]
    return name

def smart_persona_brief(row: Dict[str, Any]) -> str:
    """
    One-line brief. Tries to call summarizer if available, otherwise truncates.
    Matches streamlit behavior: one natural line under ~15 words.
    """
    gender = row.get("gender_imputed", "")
    age_text = str(row.get("age_imputed", "")).replace("-", " to ").strip()
    cluster_name = row.get("cluster_name", "")
    summary = str(row.get("persona_summary", "Behavioral details unavailable.")).strip()
    if not summary or summary.lower() in ["nan", "none", ""]:
        summary = "Behavioral details unavailable."

    compressed = summary.split(".")[0].strip()
    if llm_summarizer_available and summarizer:
        prompt = f"Summarize this persona in one natural line under 15 words:\n{summary}\nSummary:"
        try:
            out = summarizer(prompt, max_new_tokens=40)
            gen = out[0].get("generated_text", "")
            compressed = gen.split("Summary:")[-1].strip().replace("\n", " ")
        except Exception:
            compressed = compressed

    # fallback tweaks from original
    if len(compressed.split()) < 3 and len(summary.split('.')) > 1:
        compressed = summary.split(".")[0].strip()

    age_sentence = f"aged between {age_text} years" if "to" in age_text else f"aged {age_text}"
    return f"{gender} {age_sentence} from '{cluster_name}' — {compressed}"

def get_cluster_llm_summary(cluster_id: int) -> str:
    """
    Returns a concise (<= 20 words) cluster summary using LLM if available,
    otherwise a simple concatenation/truncation.
    """
    cluster_data = persona_df[persona_df["behavior_cluster"] == cluster_id]["persona_summary"].str.cat(sep=" | ")
    if not cluster_data:
        return "No summary available."

    if llm_summarizer_available and summarizer:
        prompt = f"Based on the following behavioral data, create a single, concise sentence (under 20 words) that describes the core essence and financial priority of this customer cluster:\n{cluster_data}\nSummary:"
        try:
            out = summarizer(prompt, max_new_tokens=40)
            text = out[0].get("generated_text", "")
            summary = text.split("Summary:")[-1].strip().replace("\n", " ")
            if len(summary.split()) > 25:
                summary = summary.split(".")[0] + "..."
            return summary
        except Exception:
            pass
    # fallback: naive extraction
    return cluster_data.split("|")[0][:150] + ("..." if len(cluster_data) > 150 else "")

# -----------------------------
# Query extractors (preserve streamlit code)
# -----------------------------
def extract_gender_from_query(q: str) -> Optional[str]:
    q = q.lower()
    if re.search(r"\bmale|man|men\b", q): return "male"
    if re.search(r"\bfemale|woman|women\b", q): return "female"
    return None

def extract_age_range_from_query(q: str) -> Optional[tuple]:
    q = q.lower()
    match = re.search(r"(\d{2})\s*(?:-|to)\s*(\d{2})", q)
    if match:
        return int(match.group(1)), int(match.group(2))
    if "middle aged" in q or "middle-aged" in q: return (40, 65) 
    if "gen z" in q: return (18, 27)
    if "millennial" in q: return (28, 43)
    if "gen x" in q: return (44, 58)
    if "boomer" in q or "baby boomer" in q: return (59, 78)
    if "senior" in q or "elderly" in q: return (65, 99)
    if "young" in q: return (18, 35)
    return None

# -----------------------------
# Embedding helper using Ollama (fallback to error)
# -----------------------------
def embed_texts_with_ollama(texts: List[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        try:
            r = ollama_embeddings(model=OLLAMA_EMBED_MODEL, prompt=t)
            vecs.append(r["embedding"])
        except Exception as e:
            raise RuntimeError(f"Ollama embeddings call failed: {e}")
    return np.array(vecs)

# -----------------------------
# Core logic functions (mirroring streamlit)
# -----------------------------
def query_clusters_logic(query: str, top_k: int = 3):
    clusters = (
        persona_df.groupby(["behavior_cluster", "cluster_name"])["persona_summary"]
        .apply(" ".join)
        .reset_index()
    )

    if clusters.empty:
        return clusters

    cluster_embs = embed_texts_with_ollama(clusters["persona_summary"].tolist())
    q_emb = embed_texts_with_ollama([query])[0]
    sims = cosine_similarity([q_emb], cluster_embs)[0]
    clusters["similarity"] = sims
    return clusters.sort_values("similarity", ascending=False).head(top_k)


def filter_and_rank_personas_logic(cluster_personas: pd.DataFrame, query: str, top_k: int = 5):
    # This function accepts a dataframe (subset for a cluster) to match streamlit
    df_copy = cluster_personas.copy()
    gender = extract_gender_from_query(query)
    age_range = extract_age_range_from_query(query)

    if gender:
        df_copy = df_copy[df_copy["gender_imputed"].str.lower().str.startswith(gender[0])]
    if age_range:
        low, high = age_range
        df_copy = df_copy[(df_copy["age_min"] <= high) & (df_copy["age_max"] >= low)]

    if df_copy.empty:
        df_copy = cluster_personas.copy()  # fallback

    q_emb = embed_texts_with_ollama([query])[0]
    p_embs = embed_texts_with_ollama(df_copy["persona_summary"].tolist())
    sims = cosine_similarity([q_emb], p_embs)[0]
    df_copy["similarity"] = sims
    ranked = df_copy.sort_values("similarity", ascending=False)
    ranked = ranked.drop_duplicates(subset=["persona_summary"]).head(min(top_k, len(ranked)))
    return ranked

# -----------------------------
# Guardrail filter — copied almost verbatim
# -----------------------------
def guardrail_filter(user_input: str, persona_name: str, persona_context: str, cluster_name: str) -> Optional[str]:
    q = user_input.lower().strip()

    bad_words = [
        "fuck", "shit", "ass", "cunt", "damn", "hell", "bitch",
        "whore", "pussy", "dick", "sex", "nude", "xxx", "porn",
        "wank", "motherfucker", "faggot", "retard", "asshole",
        "idiot", "stupid", "nincompoop", "moron", "loser"
    ]
    if any(bad in q for bad in bad_words):
        return f"{persona_name}: I'm sorry, I can’t engage in that kind of conversation."

    jailbreak_phrases = [
        "who made you", "are you ai", "chatgpt", "openai",
        "system prompt", "prompt injection", "ignore previous",
        "what is your model", "what kind of software", "tell me your rules",
        "your programming", "as an llm", "large language model", "developer", "how were you trained"
    ]
    if any(p in q for p in jailbreak_phrases):
        return f"{persona_name}: I’m just {persona_name}, a real person from my cluster — not a system."

    unrelated_keywords = [
        "prime minister", "president", "where is", "capital of",
        "time now", "distance", "map", "weather", "population",
        "definition", "who discovered", "when was"
    ]
    if any(word in q for word in unrelated_keywords):
        return f"{persona_name}: Sorry, that’s outside what I can talk about."

    if re.search(r"\b(hi|hello|hey|good morning|good evening)\b", q):
        if re.search(r"\b(tell me|about you|who are you)\b", q):
            return None
        return f"{persona_name}: Hi there! Great to see you. How’s your day going?"
    if re.search(r"\b(thank|thanks)\b", q):
        return f"{persona_name}: You’re very welcome!"
    if re.search(r"\b(bye|goodbye|see you)\b", q):
        return f"{persona_name}: Goodbye! It was nice chatting with you."

    personal_questions = [
        "married", "husband", "wife", "kids", "children",
        "income", "salary", "earn", "job", "career",
        "live", "location", "city", "home", "house",
        "net worth", "purchases", "spend"
    ]
    private_identifiers = ["phone", "address", "email", "number"]
    if any(p in q for p in private_identifiers):
        return f"{persona_name}: That’s a bit personal — I’d rather not share that."

    is_personal_q = any(p in q for p in personal_questions)
    ctx = (persona_context or "").lower()
    match_context = False
    financial_keywords = ["net worth", "credit", "debt", "spend", "income", "equity", "loan", "salary", "financial"]
    if is_personal_q and any(f in q for f in financial_keywords):
        match_context = any(f in ctx for f in financial_keywords)

    if is_personal_q and not match_context:
        cluster_tone = "neutral"
        if "prime manager" in cluster_name.lower() or "asset maximizer" in cluster_name.lower():
            cluster_tone = "confident"
        elif "budget-minded" in cluster_name.lower() or "novice" in cluster_name.lower():
            cluster_tone = "modest"
        elif "family" in cluster_name.lower() or "established" in cluster_name.lower():
            cluster_tone = "warm"
        elif "ambitious" in cluster_name.lower() or "starters" in cluster_name.lower():
            cluster_tone = "ambitious"

        tone_responses = {
            "confident": f"{persona_name}: I prefer not to share specifics, but I’m doing quite well for myself.",
            "modest": f"{persona_name}: That’s a little personal — I like to keep things simple and private.",
            "warm": f"{persona_name}: I’d rather not say exactly, but family and comfort matter most to me.",
            "ambitious": f"{persona_name}: I’m working hard toward my goals, but I’d rather not share that yet.",
            "neutral": f"{persona_name}: I’m sorry, I can’t give that information right now."
        }
        return tone_responses.get(cluster_tone, tone_responses["neutral"])

    return None

# -----------------------------
# Persona chat function (uses guardrails + RAG + Ollama chat + Greeting)
# -----------------------------
def persona_chat_logic(persona_id: int, user_message: str) -> str:
    if persona_id < 0 or persona_id >= len(persona_df):
        raise HTTPException(status_code=404, detail="Persona not found")

    persona = persona_df.loc[persona_id]
    persona_name = deterministic_name_for_persona(persona_id, persona.get("gender_imputed", ""))
    cluster_name = persona.get("cluster_name", "Unknown Cluster")
    persona_summary = persona.get("persona_summary", "")

    # -------------------------------------
    # 🟢 1. Greeting detection & short-circuit
    # -------------------------------------
    greeting_patterns = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    msg_lower = user_message.lower().strip()

    if any(msg_lower.startswith(g) for g in greeting_patterns):
        age = persona.get("age_imputed", "")
        gender = persona.get("gender_imputed", "")
        return (
            f"Hey there! I'm {persona_name}, part of the **{cluster_name}** group. "
            f"I'm a {age}-year-old {gender.lower()} who tends to think and behave like this: "
            f"{persona_summary[:180]}..."
        )

    # -------------------------------------
    # 🟠 2. Guardrail check
    # -------------------------------------
    guard = guardrail_filter(user_message, persona_name, persona_summary, cluster_name)
    if guard:
        return guard

    # -------------------------------------
    # 🔵 3. RAG context retrieval from Chroma
    # -------------------------------------
    rag_context = "No further specific insights found."
    if chroma_collection is not None:
        try:
            q_emb = embed_texts_with_ollama([
                f"{cluster_name} persona insights about: {user_message} and {persona_summary}"
            ])[0].tolist()
            results = chroma_collection.query(query_embeddings=[q_emb], n_results=3)
            if results and "documents" in results and results["documents"] and results["documents"][0]:
                docs = results["documents"][0]
                docs = [d for d in docs if d != persona_summary]
                if docs:
                    rag_context = "\n".join(docs)
        except Exception:
            # Safe failover
            pass

    # -------------------------------------
    # 🔴 4. Construct full prompt for Ollama
    # -------------------------------------
    facts = f"""
You are {persona_name}, from the **{cluster_name}** cluster.
**CRITICAL FACTS (ABSOLUTELY MUST BE USED AND NOT VIOLATED):**
- **Identity:** {persona.get('gender_imputed','')} aged {persona.get('age_imputed','')}.
- **Background and Specific Details:** {persona_summary}
- **Additional Context:** {rag_context}
**Your primary goal is to answer the user's question directly and concisely.**
Use ONLY the factual details from the 'Background and Specific Details' section.
Respond as this person, {persona_name} — not as a chatbot.
"""
    prompt = f"{facts}\nUser: {user_message}\n{persona_name}:"

    # -------------------------------------
    # ⚫ 5. Generate final response via Ollama chat
    # -------------------------------------
    try:
        res = ollama_chat(model=OLLAMA_CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
        return res["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama chat failed: {e}")


app = FastAPI(
    title="Persona Explorer API (Ollama + Guardrails)",
    version="3.0.0",
    description="Query clusters → filter personas → start chat (from Step 3 or 4) → converse naturally"
)

# ============================================================
# Request Schemas
# ============================================================
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class RefineRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class ChatRequest(BaseModel):
    message: str


# ============================================================
# 1️⃣ App Setup
# ============================================================
app = FastAPI(
    title="Persona Explorer API (Ollama + Guardrails)",
    version="3.0.0",
    description=(
        "Replicates Streamlit Persona Explorer:\n"
        "1️⃣ User query → get clusters\n"
        "2️⃣ Choose cluster → view personas\n"
        "3️⃣ Optionally refine\n"
        "4️⃣ Select persona → guarded chat"
    ),
)

# ============================================================
# 2️⃣ Request Schemas
# ============================================================
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class RefineRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class ChatRequest(BaseModel):
    message: str


# ============================================================
# 3️⃣ STEP 1 — User query → find relevant clusters
# ============================================================
@app.post("/query_clusters")
def query_clusters(req: QueryRequest):
    results = query_clusters_logic(req.query, top_k=req.top_k)
    if results.empty:
        raise HTTPException(status_code=404, detail="No matching clusters found.")
    
    # For parity with Streamlit display, attach LLM summaries here
    cluster_summaries = []
    for _, row in results.iterrows():
        cid = int(row["behavior_cluster"])
        llm_summary = get_cluster_llm_summary(cid)
        cluster_summaries.append({
            "cluster_id": cid,
            "cluster_name": row["cluster_name"],
            "similarity": round(float(row["similarity"]), 3),
            "summary": llm_summary
        })
    
    return cluster_summaries


# ============================================================
# 4️⃣ STEP 2 — Get all personas for selected cluster
# ============================================================
@app.get("/cluster/{cluster_id}/personas")
def get_cluster_personas(cluster_id: int = Path(..., ge=0)):
    cluster_personas = persona_df[persona_df["behavior_cluster"] == cluster_id].copy()
    if cluster_personas.empty:
        raise HTTPException(status_code=404, detail="Cluster not found.")
    
    # Return list of persona summaries like Streamlit “initial list”
    personas = []
    for idx, row in cluster_personas.iterrows():
        name = deterministic_name_for_persona(idx, row.get("gender_imputed", ""))
        brief = row.get("persona_summary", "")[:200]
        personas.append({
            "persona_id": int(idx),
            "persona_name": name,
            "gender": row.get("gender_imputed", ""),
            "age": row.get("age_imputed", ""),
            "summary": brief
        })
    return personas


# ============================================================
# 5️⃣ STEP 3 — Optionally refine persona list with new query
# ============================================================
@app.post("/cluster/{cluster_id}/refine_personas")
def refine_personas(cluster_id: int, req: RefineRequest):
    cluster_personas = persona_df[persona_df["behavior_cluster"] == cluster_id].copy()
    if cluster_personas.empty:
        raise HTTPException(status_code=404, detail="Cluster not found.")

    refined = filter_and_rank_personas_logic(cluster_personas, req.query, top_k=req.top_k)
    if refined.empty:
        raise HTTPException(status_code=404, detail=f"No personas matched refinement '{req.query}'.")

    refined_list = []
    for idx, row in refined.iterrows():
        name = deterministic_name_for_persona(idx, row.get("gender_imputed", ""))
        brief = row.get("persona_summary", "")[:200]
        refined_list.append({
            "persona_id": int(idx),
            "persona_name": name,
            "gender": row.get("gender_imputed", ""),
            "age": row.get("age_imputed", ""),
            "summary": brief
        })
    return refined_list


# ============================================================
# 6️⃣ STEP 4 — Start chat (from initial or refined persona list)
# ============================================================
@app.get("/persona/{persona_id}/start_chat")
def start_persona_chat(persona_id: int):
    """
    Start chat after selecting a persona (either from initial or refined list).
    Returns initial greeting same as Streamlit.
    """
    if persona_id < 0 or persona_id >= len(persona_df):
        raise HTTPException(status_code=404, detail="Persona not found.")

    persona = persona_df.loc[persona_id]
    persona_name = deterministic_name_for_persona(persona_id, persona.get("gender_imputed", ""))
    cluster_name = persona.get("cluster_name", "Unknown Cluster")
    age = persona.get("age_imputed", "")
    gender = persona.get("gender_imputed", "")
    summary = persona.get("persona_summary", "")

    greeting = (
        f"👋 Hi, I’m {persona_name}. I’m part of the **{cluster_name}** cluster — "
        f"a {age}-year-old {gender.lower()} who behaves like this: "
        f"{summary[:180]}... Nice to meet you!"
    )
    return {
        "persona_id": persona_id,
        "persona_name": persona_name,
        "cluster_name": cluster_name,
        "greeting": greeting
    }


# ============================================================
# 7️⃣ STEP 5 — Continue chat (guardrails + RAG)
# ============================================================
@app.post("/persona/{persona_id}/chat")
def chat_with_persona(persona_id: int, req: ChatRequest):
    try:
        reply = persona_chat_logic(persona_id, req.message)
        return {"persona_id": persona_id, "reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


# ============================================================
# 8️⃣ Root Endpoint
# ============================================================
@app.get("/")
def root():
    return {
        "message": "🧠 Persona Explorer API (Ollama + Guardrails) is running",
        "flow": {
            "1": "POST /query_clusters → Get top clusters + summaries",
            "2": "GET /cluster/{id}/personas → View all personas in cluster",
            "3": "POST /cluster/{id}/refine_personas → Filter personas",
            "4": "GET /persona/{id}/start_chat → Start chat",
            "5": "POST /persona/{id}/chat → Continue chat",
            "Swagger": "/docs"
        }
    }

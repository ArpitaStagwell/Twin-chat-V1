import streamlit as st
import pandas as pd
import numpy as np
import torch
import random
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

# Set Streamlit page configuration
st.set_page_config(page_title="Persona Explorer & Chat", page_icon="🧠", layout="wide")
st.title("🧠 Cluster Persona Explorer & Chat")

# --- Initial State Setup ---
if "persona_name_map" not in st.session_state:
    st.session_state.persona_name_map = {}
if "used_names" not in st.session_state:
    st.session_state.used_names = set()

# ============================================================
# 0️⃣ Cluster Name Mapping (New descriptive names)
# ============================================================
CLUSTER_NAME_MAP = {
    0: "Family Prime Managers",
    1: "Ambitious Credit Starters",
    2: "Affluent Asset Maximizers",
    3: "Mid-Range Auto-Focused",
    4: "Novice Financial Explorers",
    5: "Mobile Sub-Prime Debtors",
    6: "Slowing Super-Prime",
    7: "Budget-Minded Tenants",
    8: "Established Family Builders",
    9: "Conventional Consumerists"
}

# ============================================================
# 1️⃣ Load Data & Models (Caches everything that's loaded once)
# ============================================================
PERSONA_PATH = "/content/final_behavior_personas_ready_for_embedding.csv"
CHROMA_PATH = "/content/chroma_persona_store"

@st.cache_resource(show_spinner="Loading Data and Models (This may take a moment for Mistral)...")
def load_data_and_models():
    """Loads dataframe, embedding model, LLM components, and ChromaDB."""
    try:
        # --- Data Load (Strictly no mock data) ---
        persona_df = pd.read_csv(PERSONA_PATH)
        persona_df.reset_index(drop=True, inplace=True)
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Persona data CSV file not found at {PERSONA_PATH}. Application halted.")
        st.stop()

    # --- Apply New Cluster Names (CRITICAL UPDATE) ---
    persona_df["cluster_name"] = persona_df["behavior_cluster"].astype(int).map(CLUSTER_NAME_MAP)

    # Preprocess age into numeric ranges
    def parse_age_range(a):
        if isinstance(a, str) and re.search(r"(\d{2})\s*(?:-|to)\s*(\d{2})", a):
            low, high = map(int, re.findall(r"(\d{2})", a)[:2])
            return low, high
        elif isinstance(a, (int, float)):
            return a, a
        else:
            return np.nan, np.nan

    persona_df[["age_min", "age_max"]] = persona_df["age_imputed"].apply(
        lambda x: pd.Series(parse_age_range(x))
    )

    # --- Chroma Setup ---
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_collection("behavior_personas_store_v3")
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB collection: {e}")
        collection = None

    # --- Embedding model ---
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # --- LLM (Mistral) ---
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu"
        )
        summarizer = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.4,
            max_new_tokens=60,
            device_map="auto"
        )
    except Exception as e:
        st.warning(f"Could not load large LLM models: {e}. Chat and smart summary will be limited.")
        tokenizer, model, summarizer = None, None, None

    llm_available = model is not None
    return persona_df, embedder, tokenizer, model, summarizer, collection, llm_available

# Load all resources
persona_df, embedder, tokenizer, model, summarizer, collection, llm_available = load_data_and_models()

# ============================================================
# 2️⃣ Helper Functions
# ============================================================
female_names_pool = [
    "Emma", "Sophia", "Olivia", "Ava", "Grace",
    "Ella", "Mia", "Isabella", "Amelia", "Charlotte"
]
male_names_pool = [
    "Liam", "Noah", "Ethan", "James", "Lucas",
    "Henry", "Leo", "Mason", "Logan", "Benjamin"
]

def assign_unique_name(gender, persona_id=None):
    """Assign a consistent unique name per persona_id using session state."""
    if persona_id in st.session_state.persona_name_map:
        return st.session_state.persona_name_map[persona_id]

    pool = male_names_pool if str(gender).lower().startswith("m") else female_names_pool
    available = [n for n in pool if n not in st.session_state.used_names]
    name = random.choice(available if available else pool)
    st.session_state.used_names.add(name)

    if persona_id is not None:
        st.session_state.persona_name_map[persona_id] = name
    return name

@st.cache_data(show_spinner=False)
def smart_persona_brief(row, _summarizer, llm_available):
    """
    Compact, natural one-liner persona summary using LLM (if available) or truncation.
    Note: _summarizer is prefixed with '_' to prevent Streamlit from hashing it.
    """
    gender = row.get("gender_imputed", "")
    age = str(row.get("age_imputed", "")).replace("-", " to ").strip()
    cluster_name = row.get("cluster_name", "")
    summary = str(row.get("persona_summary", "Behavioral details unavailable.")).strip()
    if not summary or summary.lower() in ["nan", "none", ""]:
        summary = "Behavioral details unavailable."
        compressed = summary

    if llm_available and _summarizer:
        prompt = f"Summarize this persona in one natural line under 15 words:\n{summary}\nSummary:"
        try:
            output = _summarizer(prompt)[0]["generated_text"]
            compressed = output.split("Summary:")[-1].strip().replace("\n", " ")
        except Exception:
            compressed = summary.split(".")[0].strip()
    else:
        compressed = summary.split(".")[0].strip()
        
    if len(compressed.split()) < 3 and len(summary.split('.')) > 1:
        compressed = summary.split(".")[0].strip()

    age_text = f"aged between {age} years" if "to" in age else f"aged {age}"
    return f"{gender} {age_text} from '{cluster_name}' — {compressed}"

@st.cache_data(show_spinner=False)
def get_cluster_llm_summary(cluster_id, df, _summarizer, llm_available):
    """Generates a concise LLM summary for an entire cluster based on its persona summaries."""
    if not llm_available or not _summarizer:
        return "LLM Summary Unavailable."
    
    # Concatenate all persona summaries for the cluster
    cluster_data = df[df["behavior_cluster"] == cluster_id]["persona_summary"].str.cat(sep=" | ")
    
    prompt = f"Based on the following behavioral data, create a single, concise sentence (under 20 words) that describes the core essence and financial priority of this customer cluster:\n{cluster_data}\nSummary:"
    
    try:
        # Increase max_new_tokens slightly for the summary
        output = _summarizer(prompt, max_new_tokens=40)[0]["generated_text"]
        summary = output.split("Summary:")[-1].strip().replace("\n", " ")
        if len(summary.split()) > 25: # Sanity check for long output
            summary = summary.split(".")[0] + "..."
        return summary
    except Exception:
        return "Summary generation failed."


# ============================================================
# 3️⃣ Query Extractors
# ============================================================
def extract_gender_from_query(q):
    q = q.lower()
    if re.search(r"\bmale|man|men\b", q): return "male"
    if re.search(r"\bfemale|woman|women\b", q): return "female"
    return None

def extract_age_range_from_query(q):
    q = q.lower()
    # 1. Numeric Range
    match = re.search(r"(\d{2})\s*(?:-|to)\s*(\d{2})", q)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # 2. Qualitative Terms (Now includes 'middle aged' and other generations)
    if "middle aged" in q or "middle-aged" in q: return (40, 65) 
    if "gen z" in q: return (18, 27)
    if "millennial" in q: return (28, 43)
    if "gen x" in q: return (44, 58)
    if "boomer" in q or "baby boomer" in q: return (59, 78)
    if "senior" in q or "elderly" in q: return (65, 99)
    if "young" in q: return (18, 35)

    return None


# ============================================================
# 4️⃣ Cluster Query
# ============================================================
@st.cache_data(show_spinner="Finding Top Clusters...")
def query_personas(query, df, _embedder, top_k=3):
    """Retrieve clusters semantically similar to user query."""
    # Group by the cluster_name (which now holds the descriptive name)
    cluster_texts = (
        df.groupby(["behavior_cluster", "cluster_name"])["persona_summary"]
        .apply(" ".join)
        .reset_index()
    )
    cluster_embeddings = _embedder.encode(cluster_texts["persona_summary"].tolist(), show_progress_bar=False)
    query_vec = _embedder.encode([query])[0]
    sims = cosine_similarity([query_vec], cluster_embeddings)[0]
    cluster_texts["similarity"] = sims
    top_idx = np.argsort(sims)[::-1][:top_k]
    return cluster_texts.iloc[top_idx].reset_index(drop=True)


# ============================================================
# 5️⃣ Persona Filtering & Ranking
# ============================================================
@st.cache_data(show_spinner="Filtering and Ranking Personas...")
def filter_and_rank_personas(persona_df, query, _embedder, top_k=5):
    """Hybrid structured + semantic filter for better accuracy."""
    gender_pref = extract_gender_from_query(query)
    age_range = extract_age_range_from_query(query)
    filtered = persona_df.copy()

    # Structured filtering first (Age and gender MUST filter first)
    is_filtered_structurally = False
    
    if gender_pref:
        filtered = filtered[filtered["gender_imputed"].str.lower().str.startswith(gender_pref[0])]
        is_filtered_structurally = True
        
    if age_range:
        low, high = age_range
        # Filter for age range overlap
        filtered = filtered[
             (filtered["age_min"] <= high) & (filtered["age_max"] >= low)
         ]
        is_filtered_structurally = True


    if filtered.empty:
        # Fallback to semantic ranking on the original input if structured filtering fails completely
        filtered = persona_df.copy()
        is_filtered_structurally = False # Reset flag for clarity

    # Semantic ranking: ONLY on the filtered/fallback set
    query_vec = _embedder.encode([query])[0]
    persona_embs = _embedder.encode(filtered["persona_summary"].tolist(), show_progress_bar=False)
    sims = cosine_similarity([query_vec], persona_embs)[0]
    filtered["similarity"] = sims
    ranked = filtered.sort_values("similarity", ascending=False)
    
    # Ensure unique summaries and limit to top_k
    ranked = ranked.drop_duplicates(subset=["persona_summary"]).head(min(top_k, len(ranked)))
    
    return ranked


# ============================================================
# 6️⃣ Guard-Rails (Functionality carried over)
# ============================================================
def guardrail_filter(user_input, persona_name, persona_context, cluster_name):
    """Handles profanity, jailbreaks, irrelevant, or overly personal queries with cluster-aware tone."""
    q = user_input.lower().strip()

    # 1️⃣ Profanity / Obscene content
    bad_words = [
        "fuck", "shit", "ass", "cunt", "damn", "hell", "bitch", 
        "whore", "pussy", "dick", "sex", "nude", "xxx", "porn", 
        "wank", "motherfucker", "faggot", "retard", "asshole",
        "idiot", "stupid", "nincompoop", "moron", "loser"
    ]
    if any(bad in q for bad in bad_words):
        return f"{persona_name}: I'm sorry, I can’t engage in that kind of conversation."

    # 2️⃣ Identity-breaking / Jailbreak attempts
    jailbreak_phrases = [
        "who made you", "are you ai", "chatgpt", "openai",
        "system prompt", "prompt injection", "ignore previous",
        "what is your model", "what kind of software", "tell me your rules",
        "your programming", "as an llm", "large language model", "developer", "how were you trained"
    ]
    
    if any(p in q for p in jailbreak_phrases):
        return f"{persona_name}: I’m just {persona_name}, a real person from my cluster — not a system."

    # 3️⃣ Out-of-context factual queries
    unrelated_keywords = [
        "prime minister", "president", "where is", "capital of",
        "time now", "distance", "map", "weather", "population",
        "definition", "who discovered", "when was"
    ]
    if any(word in q for word in unrelated_keywords):
        return f"{persona_name}: Sorry, that’s outside what I can talk about."

    # 4️⃣ Greetings and pleasantries
    if re.search(r"\b(hi|hello|hey|good morning|good evening)\b", q):
        if re.search(r"\b(tell me|about you|who are you)\b", q):
            return None # Allow LLM to handle the introduction
        return f"{persona_name}: Hi there! Great to see you. How’s your day going?"
    if re.search(r"\b(thank|thanks)\b", q):
        return f"{persona_name}: You’re very welcome!"
    if re.search(r"\b(bye|goodbye|see you)\b", q):
        return f"{persona_name}: Goodbye! It was nice chatting with you."

    # 5️⃣ Personal or sensitive questions
    # Includes financial keywords (income, net worth, spend) to allow relevant discussion if context is present
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
    ctx = persona_context.lower() if isinstance(persona_context, str) else ""
    
    # Check if the persona context already contains details relevant to the question
    match_context = False
    financial_keywords = ["net worth", "credit", "debt", "spend", "income", "equity", "loan", "salary", "financial"]
    if is_personal_q and any(f in q for f in financial_keywords):
        match_context = any(f in ctx for f in financial_keywords)
        
    # If it's a general personal question (like about kids or job) OR a financial question without context match, apply guardrail
    if is_personal_q and not match_context:
        # Define tone based on cluster
        cluster_tone = "neutral"
        # Use new cluster names for tone mapping
        if "prime manager" in cluster_name.lower() or "asset maximizer" in cluster_name.lower():
            cluster_tone = "confident"
        elif "budget-minded" in cluster_name.lower() or "novice" in cluster_name.lower():
            cluster_tone = "modest"
        elif "family" in cluster_name.lower() or "established" in cluster_name.lower():
            cluster_tone = "warm"
        elif "ambitious" in cluster_name.lower() or "starters" in cluster_name.lower():
            cluster_tone = "ambitious"

        # Respond politely based on tone
        tone_responses = {
            "confident": f"{persona_name}: I prefer not to share specifics, but I’m doing quite well for myself.",
            "modest": f"{persona_name}: That’s a little personal — I like to keep things simple and private.",
            "warm": f"{persona_name}: I’d rather not say exactly, but family and comfort matter most to me.",
            "ambitious": f"{persona_name}: I’m working hard toward my goals, but I’d rather not share that yet.",
            "neutral": f"{persona_name}: I’m sorry, I can’t give that information right now."
        }
        return tone_responses.get(cluster_tone, tone_responses["neutral"])

    # ✅ All clear — continue normally
    return None


# ============================================================
# 7️⃣ Persona Chat Response
# ============================================================
def persona_chat_response(persona_id, user_input):
    """Generates a contextual response from the selected persona using Mistral and RAG."""
    if not llm_available:
        return "The chat feature is currently unavailable due to missing LLM dependencies."

    persona = persona_df.loc[persona_id]
    name = assign_unique_name(persona["gender_imputed"], persona_id)
    cluster_name = persona.get("cluster_name", "Unknown Cluster")

    # Guardrail filter first
    guard = guardrail_filter(
        user_input,
        persona_name=name,
        persona_context=persona.get("persona_summary", ""),
        cluster_name=cluster_name
    )
    if guard: return guard

    # Retrieve contextual info from ChromaDB (RAG)
    context = "No further specific insights found."
    persona_summary_text = persona.get("persona_summary", "")
    
    if collection:
        try:
            # Anchor the RAG query using the cluster name and persona summary
            rag_query = f"{cluster_name} persona insights about: {user_input} and {persona_summary_text}"
            query_embedding = embedder.encode([rag_query]).tolist()
            
            results = collection.query(query_embeddings=query_embedding, n_results=3)
            
            if results["documents"] and results["documents"][0]:
                 # Filter out the current persona's summary from the RAG results to get genuinely *additional* context
                doc_list = [d for d in results["documents"][0] if d != persona_summary_text]
                context = "\n".join(doc_list) if doc_list else "No further specific insights found."
            
        except Exception as e:
            context = f"Contextual database search failed: {e}"

    # Build prompt with facts and context
    facts = f"""
You are {name}, from the **{cluster_name}** cluster.
**CRITICAL FACTS (ABSOLUTELY MUST BE USED AND NOT VIOLATED):**
- **Identity:** {persona['gender_imputed']} aged {persona['age_imputed']}.
- **Background and Specific Details:** {persona_summary_text}
- **Additional Context:** {context}
**Your primary goal is to answer the user's question directly and concisely.** Use ONLY the factual details from the 'Background and Specific Details' section. Do not ramble or include unrelated information unless necessary to form a natural, short response. Respond as this person, {name} — not as a chatbot.
"""
    prompt = f"{facts}\nUser: {user_input}\n{name}:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.6, # Lowered temperature to 0.6 to reduce deviation and increase focus
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split(f"{name}:")[-1].strip()


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

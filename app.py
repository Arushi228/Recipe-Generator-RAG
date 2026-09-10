import os
import re
import threading
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

RECIPES_FILE = os.path.join(os.path.dirname(__file__), "recipes_data.txt")
INDEX_DIR    = os.path.join(os.path.dirname(__file__), "faiss_index")

# ─────────────────────────────────────────────
# RAG: Build FAISS vector store
# ─────────────────────────────────────────────
def build_vector_store():
    # Manually split on --- so each Document = exactly one full recipe
    with open(RECIPES_FILE, encoding="utf-8") as f:
        raw = f.read()

    from langchain.schema import Document
    recipe_blocks = [b.strip() for b in re.split(r"\n---+\n", raw) if b.strip() and "RECIPE:" in b]
    chunks = [Document(page_content=block) for block in recipe_blocks]
    print(f"[RAG] Loaded {len(chunks)} recipe chunks from knowledge base.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(INDEX_DIR):
        print("[RAG] Loading existing FAISS index...")
        vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        print("[RAG] Building new FAISS index...")
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(INDEX_DIR)
        print(f"[RAG] FAISS index saved to {INDEX_DIR}")

    print("[RAG] Vector store ready.")
    return vs


# ─────────────────────────────────────────────
# Recipe formatter  (the "generation" step)
# ─────────────────────────────────────────────
def parse_recipe_block(text: str) -> dict:
    """Parse a raw recipe text block into structured fields."""
    def field(pattern, default=""):
        """Extract a single-line field — MULTILINE so ^ matches each line."""
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else default

    def block(pattern, default=""):
        """Extract a multi-line block (DOTALL, non-greedy)."""
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else default

    name         = field(r"^RECIPE:\s*(.+)",     "")
    category     = field(r"^CATEGORY:\s*(.+)",   "")
    prep         = field(r"^PREP TIME:\s*(.+)",  "")
    cook         = field(r"^COOK TIME:\s*(.+)",  "")
    servings     = field(r"^SERVINGS:\s*(.+)",   "")
    difficulty   = field(r"^DIFFICULTY:\s*(.+)", "")
    tips         = field(r"^TIPS:\s*(.+)",       "")
    nutrition    = field(r"^NUTRITION:\s*(.+)",  "")

    # Multi-line blocks — non-greedy, stop at the next section header
    ing_block = block(r"INGREDIENTS:\s*\n(.*?)(?=\nINSTRUCTIONS:)", "")
    ingredients = [l.lstrip("- ").strip() for l in ing_block.splitlines()
                   if l.strip().startswith("-")]

    ins_block = block(r"INSTRUCTIONS:\s*\n(.*?)(?=\nTIPS:|\nNUTRITION:|$)", "")
    instructions = [l.strip() for l in ins_block.splitlines()
                    if re.match(r"^\d+\.", l.strip())]

    return {
        "name": name, "category": category, "prep": prep, "cook": cook,
        "servings": servings, "difficulty": difficulty,
        "ingredients": ingredients, "instructions": instructions,
        "tips": tips, "nutrition": nutrition,
    }


def format_recipe_answer(query: str, docs: list) -> tuple[str, list]:
    """
    Build a rich, formatted answer from the retrieved recipe documents.
    Returns (markdown_string, source_names_list).
    """
    query_lower = query.lower()
    sources = []

    # Each FAISS doc may contain multiple recipes (chunk may span ---).
    # Merge all docs then split cleanly on the --- separator.
    full_text = "\n---\n".join(d.page_content for d in docs)
    blocks = re.split(r"-{3,}", full_text)

    recipes = []
    for block in blocks:
        if "RECIPE:" in block:
            r = parse_recipe_block(block)
            if r["name"] and r["name"] not in sources:
                sources.append(r["name"])
                recipes.append(r)

    if not recipes:
        # Fallback: return raw chunk text nicely formatted
        return _format_raw(docs), []

    # Score recipes by query relevance
    def score(r):
        combined = (r["name"] + " " + r["category"]).lower()
        query_words = re.findall(r"\w+", query_lower)
        return sum(1 for w in query_words if w in combined)

    recipes.sort(key=score, reverse=True)
    best = recipes[0]

    # Build the formatted answer
    lines = []

    # Header
    lines.append(f"## 🍽️ {best['name']}")
    if best["category"]:
        lines.append(f"**Category:** {best['category']}")

    # Meta row
    meta = []
    if best["prep"]:      meta.append(f"⏱ Prep: {best['prep']}")
    if best["cook"]:      meta.append(f"🔥 Cook: {best['cook']}")
    if best["servings"]:  meta.append(f"👥 Serves: {best['servings']}")
    if best["difficulty"]:meta.append(f"📊 Difficulty: {best['difficulty']}")
    if meta:
        lines.append("  ".join(meta))

    lines.append("")

    # Ingredients
    if best["ingredients"]:
        lines.append("### 🛒 Ingredients")
        for ing in best["ingredients"]:
            lines.append(f"- {ing}")
        lines.append("")

    # Instructions
    if best["instructions"]:
        lines.append("### 👨‍🍳 Instructions")
        for step in best["instructions"]:
            lines.append(step)
        lines.append("")

    # Tips
    if best["tips"]:
        lines.append(f"### 💡 Chef's Tip")
        lines.append(best["tips"])
        lines.append("")

    # Nutrition
    if best["nutrition"]:
        lines.append(f"### 📊 Nutrition (per serving)")
        lines.append(best["nutrition"])

    # If query asks for something not in KB, suggest similar
    if score(best) == 0:
        lines.insert(2, f"\n> I don't have that exact recipe in my knowledge base, but here's a similar one you might enjoy!\n")

    return "\n".join(lines), sources


def _format_raw(docs: list) -> str:
    """Fallback: return the raw retrieved text stripped up nicely."""
    parts = []
    for d in docs[:2]:
        parts.append(d.page_content.strip())
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────
# Initialise in background so Flask starts immediately
# ─────────────────────────────────────────────
vector_store = None
_rag_ready = False
_rag_error: str | None = None


def _init_rag():
    global vector_store, _rag_ready, _rag_error
    print("[RAG] Initialising RAG pipeline (background)...")
    try:
        vector_store = build_vector_store()
        _rag_ready = True
        print("[RAG] Pipeline ready.")
    except Exception as e:
        _rag_error = str(e)
        print(f"[RAG] STARTUP ERROR: {e}")


threading.Thread(target=_init_rag, daemon=True).start()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask_recipe():
    data = request.get_json(silent=True)
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field in request body."}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    if not _rag_ready:
        msg = f"RAG pipeline failed to start: {_rag_error}" if _rag_error else "RAG pipeline is still warming up, please retry in a moment."
        return jsonify({"error": msg, "status": "warming_up"}), 503

    try:
        docs = vector_store.similarity_search(query, k=4)
        answer, sources = format_recipe_answer(query, docs)
        return jsonify({"answer": answer, "sources": sources, "status": "success"})
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/suggest", methods=["GET"])
def suggest_recipes():
    suggestions = [
        {"name": "Classic Margherita Pizza",      "emoji": "🍕", "category": "Italian"},
        {"name": "Chicken Tikka Masala",           "emoji": "🍛", "category": "Indian"},
        {"name": "Avocado Toast with Poached Eggs","emoji": "🥑", "category": "Breakfast"},
        {"name": "Classic Beef Tacos",             "emoji": "🌮", "category": "Mexican"},
        {"name": "Creamy Mushroom Risotto",        "emoji": "🍄", "category": "Italian"},
        {"name": "Chocolate Lava Cake",            "emoji": "🍫", "category": "Dessert"},
        {"name": "Pad Thai",                       "emoji": "🍜", "category": "Thai"},
        {"name": "Greek Salad",                    "emoji": "🥗", "category": "Greek"},
        {"name": "Lemon Garlic Butter Salmon",     "emoji": "🐟", "category": "Seafood"},
        {"name": "Vegetable Curry",                "emoji": "🫘", "category": "Indian"},
    ]
    return jsonify({"suggestions": suggestions})


@app.route("/api/search", methods=["POST"])
def semantic_search():
    data = request.get_json(silent=True)
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field."}), 400

    query = data["query"].strip()
    if not _rag_ready:
        return jsonify({"error": "Vector store is still warming up, please retry in a moment."}), 503

    try:
        docs = vector_store.similarity_search(query, k=4)
        results = []
        seen = set()
        for doc in docs:
            for line in doc.page_content.split("\n"):
                if line.startswith("RECIPE:"):
                    name = line.replace("RECIPE:", "").strip()
                    if name not in seen:
                        seen.add(name)
                        results.append({"name": name, "snippet": doc.page_content[:200] + "..."})
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    if _rag_error:
        rag_status = "error"
    elif _rag_ready:
        rag_status = "active"
    else:
        rag_status = "warming_up"
    return jsonify({"status": "ok", "model": "RAG + flan-t5-base (local)", "rag": rag_status})


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)

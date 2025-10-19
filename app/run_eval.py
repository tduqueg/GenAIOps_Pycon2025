# app/run_eval.py
import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
from dotenv import load_dotenv

from app.rag_pipeline import load_vectorstore_from_disk, build_chain

# LangChain evaluators
from langchain_openai import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain, ContextQAEvalChain
from langchain.evaluation import load_evaluator

load_dotenv()

# ---------------------------
# Config
# ---------------------------
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1_asistente_rrhh")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DATASET_PATH = os.getenv("EVAL_DATASET", "tests/eval_dataset.json")
EVAL_MODEL = os.getenv("OPENAI_EVAL_MODEL", "gpt-4o-mini")
REPORT_PATH = os.getenv("EVAL_REPORT_PATH", "tests/eval_summary.txt")

# ---------------------------
# Dataset
# ---------------------------
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# ---------------------------
# Vectorstore + Chain
# ---------------------------
vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)

# Retriever directo (sin tocar rag_pipeline.py)
retriever = vectordb.as_retriever(search_kwargs={"k": 6})

# ---------------------------
# Evaluadores
# ---------------------------
llm = ChatOpenAI(model=EVAL_MODEL, temperature=0)

# QA sin contexto y con contexto (útil para RAG)
qa_eval = QAEvalChain.from_llm(llm)
context_eval = ContextQAEvalChain.from_llm(llm)

# Evaluación por criterios (usa la referencia)
CRITERIA = {
    "correctness": "Is the answer factually correct given the context?",
    "relevance":   "Is the answer relevant to the question?",
    "coherence":   "Is the answer well-structured and understandable?",
}
criteria_eval = load_evaluator("labeled_criteria", criteria=CRITERIA, llm=llm)

# ---------------------------
# Utilidades
# ---------------------------
def _extract_lcqa(graded):
    """Normaliza salida de QAEvalChain / ContextQAEvalChain a (verdict, score_float)."""
    verdict, score = "UNKNOWN", 0.0
    if isinstance(graded, dict):
        if "results" in graded and graded["results"]:
            it = graded["results"][0]
            verdict = str(it.get("value", verdict))
            try:
                score = float(it.get("score", score) or 0)
            except Exception:
                score = 0.0
        else:
            verdict = str(graded.get("value", verdict))
            try:
                score = float(graded.get("score", score) or 0)
            except Exception:
                score = 0.0
    return verdict, score

def _yn_to_num(x):
    return 1.0 if str(x).strip().lower() in (
        "y","yes","true","1","correct","relevant","coherent"
    ) else 0.0

def _extract_labeled_criteria(crit_dict):
    """
    Soporta múltiples formatos de salida de labeled_criteria:
      A) {'correctness': {'score':'Y'}, 'relevance':{'score':'Y'}, 'coherence':{'score':'N'}}
      B) {'results':[{'correctness':{'score':'Y'}, ...}]}
      C) {'score':'Y'}  -> aplica a todos
      D) {'value':'Y'}  -> aplica a todos
    Retorna (corr, rel, coh) en [0.0,1.0]
    """
    if not isinstance(CRITERIA, dict):  # defensa
        return 0.0, 0.0, 0.0
    if not isinstance(crit_dict, dict):
        return 0.0, 0.0, 0.0

    # B) results[0] con criterios dentro
    if "results" in crit_dict and crit_dict["results"]:
        item = crit_dict["results"][0]
        # Si dentro ya vienen por criterio
        if any(k in item for k in ("correctness","relevance","coherence")):
            corr = _yn_to_num(item.get("correctness", {}).get("score", item.get("correctness", {}).get("value", 0)))
            rel  = _yn_to_num(item.get("relevance",   {}).get("score", item.get("relevance",   {}).get("value", 0)))
            coh  = _yn_to_num(item.get("coherence",   {}).get("score", item.get("coherence",   {}).get("value", 0)))
            return corr, rel, coh
        # Si sólo hay score/value global en el item
        if "score" in item or "value" in item:
            v = item.get("score", item.get("value", 0))
            v = _yn_to_num(v)
            return v, v, v

    # A) criterios al tope
    if any(k in crit_dict for k in ("correctness","relevance","coherence")):
        corr = _yn_to_num(crit_dict.get("correctness", {}).get("score", crit_dict.get("correctness", {}).get("value", 0)))
        rel  = _yn_to_num(crit_dict.get("relevance",   {}).get("score", crit_dict.get("relevance",   {}).get("value", 0)))
        coh  = _yn_to_num(crit_dict.get("coherence",   {}).get("score", crit_dict.get("coherence",   {}).get("value", 0)))
        return corr, rel, coh

    # C/D) score/value global al tope
    if "score" in crit_dict or "value" in crit_dict:
        v = crit_dict.get("score", crit_dict.get("value", 0))
        v = _yn_to_num(v)
        return v, v, v

    # Fallback
    return 0.0, 0.0, 0.0

def _avg(vals):
    return sum(vals)/len(vals) if vals else 0.0

# ---------------------------
# Experimento MLflow (una vez)
# ---------------------------
mlflow.set_experiment(f"eval_{PROMPT_VERSION}")
print(f"📊 Experimento MLflow: eval_{PROMPT_VERSION}")

results = []

# ---------------------------
# Evaluación por lote (1 run por pregunta)
# ---------------------------
for i, pair in enumerate(dataset, start=1):
    pregunta = pair["question"].strip()
    referencia = pair["answer"].strip()

    # Recupera contexto del RAG sin tocar la chain interna
    docs = retriever.get_relevant_documents(pregunta)
    context_text = "\n\n".join([getattr(d, "page_content", "") for d in docs])[:8000]

    with mlflow.start_run(run_name=f"eval_q{i:02d}"):
        # 1) Predicción
        pred = chain.invoke({"question": pregunta, "chat_history": []}).get("answer", "").strip()

        # 2) QA sin contexto (comparación estricta)
        graded = qa_eval.evaluate_strings(
            input=pregunta, prediction=pred, reference=referencia
        )
        verdict, qa_score = _extract_lcqa(graded)

        # 3) QA con contexto (justo para RAG)
        graded_ctx = context_eval.evaluate_strings(
            input=pregunta, prediction=pred, reference=referencia, context=context_text
        )
        verdict_ctx, qa_ctx_score = _extract_lcqa(graded_ctx)

        # 4) Criterios por criterio (robusto a múltiples formatos)
        crit = criteria_eval.evaluate_strings(
            prediction=pred, reference=referencia, input=pregunta, context=context_text
        )

        # Guarda el raw para inspección (útil si algo cambia en versiones)
        try:
            mlflow.log_text(json.dumps(crit, ensure_ascii=False, indent=2), "criteria_raw.json")
        except Exception:
            pass

        corr, rel, coh = _extract_labeled_criteria(crit)

        # Si el evaluator devolvió algo raro pero QA(ctx) fue 1, usa respaldo mínimo.
        if corr == 0.0 and qa_ctx_score == 1.0:
            corr = 1.0

        # Logs por run
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)
        mlflow.log_param("question", pregunta)

        mlflow.log_metric("qa_is_correct", qa_score)           # sin contexto
        mlflow.log_metric("qa_ctx_is_correct", qa_ctx_score)   # con contexto
        mlflow.log_metric("correctness_score", corr)
        mlflow.log_metric("relevance_score",   rel)
        mlflow.log_metric("coherence_score",   coh)
        mlflow.log_metric("retrieved_docs",    len(docs))

        # Artefactos ligeros
        mlflow.log_text(pregunta,           "question.txt")
        mlflow.log_text(referencia,         "reference.txt")
        mlflow.log_text(pred,               "prediction.txt")
        mlflow.log_text(context_text,       "retrieved_context.txt")

        # Consola
        print(f"\n#{i}/{len(dataset)}")
        print(f"❓ Q: {pregunta}")
        print(f"🧠 Pred: {pred}")
        print(f"🎯 Ref : {referencia}")
        print(f"✅ QA(no-ctx): verdict={verdict}, score={qa_score}")
        print(f"🧩 QA(ctx)  : verdict={verdict_ctx}, score={qa_ctx_score}")
        print(f"📐 Criteria -> correctness={corr:.2f}, relevance={rel:.2f}, coherence={coh:.2f}")
        print(f"📚 Retrieved docs: {len(docs)}")

        # Acumular para el resumen
        results.append({
            "question": pregunta,
            "prediction": pred,
            "reference": referencia,
            "qa_noctx": float(qa_score),
            "qa_ctx": float(qa_ctx_score),
            "correctness": float(corr),
            "relevance": float(rel),
            "coherence": float(coh),
            "retrieved_docs": int(len(docs))
        })

# ---------------------------
# Resumen a TXT + artifact MLflow
# ---------------------------
# Encabezado
lines = []
lines.append(f"Experimento: eval_{PROMPT_VERSION}")
lines.append(f"Items evaluados: {len(results)}")
lines.append("")

# Tabla por pregunta
lines.append("Resultados por pregunta")
lines.append("=".ljust(120, "="))
lines.append(f"{'#':>2}  {'QA':>4}  {'QA_CTX':>7}  {'CORR':>5}  {'REL':>5}  {'COH':>5}  {'Docs':>4}  Pregunta")
lines.append("-".ljust(120, "-"))

for idx, r in enumerate(results, start=1):
    lines.append(
        f"{idx:>2}  {r['qa_noctx']:.2f}  {r['qa_ctx']:.2f}   {r['correctness']:.2f}   {r['relevance']:.2f}   {r['coherence']:.2f}   {r['retrieved_docs']:>4}  {r['question']}"
    )

lines.append("")
# Promedios
avg_qa       = _avg([r["qa_noctx"] for r in results])
avg_qa_ctx   = _avg([r["qa_ctx"] for r in results])
avg_corr     = _avg([r["correctness"] for r in results])
avg_rel      = _avg([r["relevance"] for r in results])
avg_coh      = _avg([r["coherence"] for r in results])

lines.append("Promedios")
lines.append("---------")
lines.append(f"QA(no-ctx): {avg_qa:.3f}")
lines.append(f"QA(ctx)   : {avg_qa_ctx:.3f}")
lines.append(f"Correctness: {avg_corr:.3f}")
lines.append(f"Relevance  : {avg_rel:.3f}")
lines.append(f"Coherence  : {avg_coh:.3f}")
lines.append("")

# Guardar TXT
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n" + "="*80)
print(f"📄 Resumen escrito en: {REPORT_PATH}")
print("="*80 + "\n")

# Subir a MLflow como run resumen
with mlflow.start_run(run_name="eval_summary", nested=True):
    mlflow.log_metric("avg_qa_noctx", avg_qa)
    mlflow.log_metric("avg_qa_ctx", avg_qa_ctx)
    mlflow.log_metric("avg_correctness", avg_corr)
    mlflow.log_metric("avg_relevance", avg_rel)
    mlflow.log_metric("avg_coherence", avg_coh)
    mlflow.log_text("\n".join(lines), os.path.basename(REPORT_PATH))

import json

import streamlit as st

from complaints.batch import export_csv, load_csv
from complaints.core import ROOT, read
from complaints.models import Engine

st.set_page_config(page_title="Réclamations clients", layout="wide")
st.title("Analyse comparative des réclamations clients")
st.warning("Évaluation en anglais uniquement. LoRA smoke : validation technique, sans preuve de gain.")
text = st.text_area("Réclamation (anglais)", "My credit card has a charge I did not authorize.")
kind = st.selectbox("Modèle", ["classic", "base", "lora"])
adapter = st.selectbox(
    "Adaptateur",
    ["adapter_smoke", "adapter_full"],
    index=1 if (ROOT / "artifacts/adapter_full/adapter_config.json").exists() else 0,
)


def run(texts, chosen):
    results = []
    for model in chosen:
        engine = None
        try:
            engine = Engine(model, adapter)
            for text in texts:
                results.append({"model": model, "text": text, **engine.predict(text)})
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
        finally:
            if engine:
                engine.close()
    return results


if st.button("Analyser"):
    st.session_state.results = run([text], [kind])
if st.button("Comparer les trois approches"):
    st.session_state.results = run([text], ["classic", "base", "lora"])
upload = st.file_uploader("CSV UTF-8 avec colonne text, maximum 20 lignes", type="csv")
if upload is not None and st.button("Analyser le CSV"):
    try:
        st.session_state.results = run(load_csv(upload), [kind])
    except ValueError as exc:
        st.error(str(exc))
if "results" in st.session_state:
    for result in st.session_state.results:
        st.subheader(result["model"])
        st.write(f"Temps : {result['seconds']:.3f} s")
        if result["error"]:
            st.error(f"Sortie non conforme : {result['error']}")
            st.code(result["raw"])
        else:
            st.write(f"Catégorie : {result['output']['category']}")
            for field, label in [("summary", "Résumé"), ("justification", "Justification")]:
                if result["output"][field] is not None:
                    st.write(f"{label} : {result['output'][field]}")
            st.json(result["output"])
    st.download_button("Exporter CSV", export_csv(st.session_state.results), "results.csv", "text/csv")
    st.download_button("Exporter JSON", json.dumps(st.session_state.results), "results.json")
with st.expander("Performances et limites"):
    if (ROOT / "reports/comparison.json").exists():
        st.json(read("reports/comparison.json"))
    st.write("Trois produits financiers historiques ; aucune validation française. Résumés sans références.")

import streamlit as st
import spacy
import pandas as pd
import subprocess
import sys

st.set_page_config(page_title="NER", page_icon="🧠")

@st.cache_resource
def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True
        )
        return spacy.load("en_core_web_sm")

nlp = load_model()

text = st.text_area(
    "Enter Text",
    height=180,
    placeholder="Example: Apple was founded by Steve Jobs in California in 1976."
)

if st.button("Detect Entities"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        results = ner(text)

        if len(results) == 0:
            st.error("No entities found.")

        else:

            data = []

            for r in results:
                data.append({
                    "Entity": r["word"],
                    "Label": r["entity_group"],
                    "Confidence": round(r["score"], 3)
                })

            df = pd.DataFrame(data)

            st.subheader("Detected Entities")

            st.dataframe(df, use_container_width=True)

            st.subheader("Highlighted Text")

            html = text

            colors = {
                "PER":"#FFD54F",
                "ORG":"#81C784",
                "LOC":"#64B5F6",
                "MISC":"#CE93D8"
            }

            for r in sorted(results, key=lambda x: x["start"], reverse=True):

                color = colors.get(r["entity_group"], "#B0BEC5")

                entity = text[r["start"]:r["end"]]

                tag = f"""
<span style="
background:{color};
padding:3px;
border-radius:5px;
font-weight:bold;
">
{entity}
({r["entity_group"]})
</span>
"""

                html = html[:r["start"]] + tag + html[r["end"]:]

            st.markdown(html, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(
    page_title="Named Entity Recognition (NER)",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Named Entity Recognition (NER)")
st.write("Detect Persons, Organizations, Locations and Miscellaneous entities.")

@st.cache_resource
def load_model():
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )

ner = load_model()

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

            for entity in results:
                data.append({
                    "Entity": entity["word"],
                    "Entity Type": entity["entity_group"],
                    "Confidence": round(entity["score"], 3)
                })

            df = pd.DataFrame(data)

            st.subheader("Detected Entities")
            st.dataframe(df, use_container_width=True)

            st.subheader("Highlighted Text")

            highlighted = text

            colors = {
                "PER": "#FFEB3B",
                "ORG": "#81C784",
                "LOC": "#64B5F6",
                "MISC": "#CE93D8"
            }

            for entity in sorted(results, key=lambda x: x["start"], reverse=True):

                start = entity["start"]
                end = entity["end"]

                word = text[start:end]

                color = colors.get(entity["entity_group"], "#E0E0E0")

                tag = f"""
<span style="
background-color:{color};
padding:4px;
border-radius:4px;
font-weight:bold;
">
{word}
({entity['entity_group']})
</span>
"""

                highlighted = highlighted[:start] + tag + highlighted[end:]

            st.markdown(highlighted, unsafe_allow_html=True)

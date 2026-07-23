import streamlit as st
import spacy
import pandas as pd
from spacy import displacy
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Named Entity Recognition",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Named Entity Recognition (NER)")
st.write("Detect Persons, Organizations, Locations, Dates and more.")

# Load Model
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

text = st.text_area(
    "Enter your text",
    height=180,
    placeholder="Example: Microsoft was founded by Bill Gates in 1975 in the United States."
)

if st.button("Detect Entities"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        doc = nlp(text)

        if len(doc.ents) == 0:
            st.error("No entities detected.")
        else:

            st.subheader("Detected Entities")

            data = []

            for ent in doc.ents:
                data.append({
                    "Entity": ent.text,
                    "Label": ent.label_,
                    "Meaning": spacy.explain(ent.label_)
                })

            df = pd.DataFrame(data)

            st.dataframe(df, use_container_width=True)

            st.subheader("Highlighted Text")

            html = displacy.render(doc, style="ent", jupyter=False)

            components.html(html, height=350, scrolling=True)

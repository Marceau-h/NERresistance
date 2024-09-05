"""MIT 2.0 - Marceau Hernandez, CERES - SORBONNE UNIVERSITE, 2024"""
import json

import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THRESHOLD = 0.0

nlp = spacy.load("fr_core_news_lg")

def extract_ners(text: str) -> list[str]:
    """
    Extracts named entities from a text
    :param text: The text to extract named entities from
    :return: A list of named entities
    """
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ("PER", "LOC")]

def extract_ners_df(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    return df[text_col].apply(extract_ners)


base_noms_path = "revue_resistants/base_noms.csv"
base_noms_df = pd.read_csv(base_noms_path, sep=";").fillna("")

base_revue_path = "revue_resistants/base_revue.csv"
base_revue_df = pd.read_csv(base_revue_path, sep=";", encoding="utf-8").fillna("")

docs = base_noms_df["string"].tolist() + base_revue_df["pg_text"].tolist()

tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2,2), min_df=2)
tfidf.fit(docs)

base_revue_df["ners"] = extract_ners_df(base_revue_df, "pg_text")

base_noms_vectors = tfidf.transform(base_noms_df["string"].tolist())

mega_struct = []
for i, row in base_revue_df.iterrows():
    sims_by_ner = []
    for ner in row["ners"]:
        vect = tfidf.transform([ner])
        sims = cosine_similarity(base_noms_vectors, vect)

        sims_upper_than_tresh = sims > THRESHOLD

        sims_upper_indices = sims_upper_than_tresh.nonzero()[0]
        sims_upper_values = sims[sims_upper_indices].flatten()
        sims_upper_in_df = base_noms_df.iloc[sims_upper_indices]

        sims_by_ner.append(
            {
                "ner": ner,
                "sims": sims_upper_values.tolist(),
                "noms": sims_upper_in_df["string"].tolist(),
                "id_sim": sims_upper_in_df["id_sim"].tolist(),
            }
        )

    mega_struct.append(
        {
            "id_pg": row["id_pg"],
            "ners": sims_by_ner,
        }
    )

with open("mega_struct.json", "w", encoding="utf-8") as f:
    json.dump(mega_struct, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()


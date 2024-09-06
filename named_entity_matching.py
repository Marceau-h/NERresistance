"""
This script extracts named entities from a text and compares them to a list of names.
It then saves the results in a JSON file.

This allows to identify which named entities are present in a text with a degree of similarity to a list of names.

The script requires the following files:
- input_names.csv: A CSV file with a column for names to compare to and an ID column
- input_texts.csv: A CSV file with a column for texts to extract named entities from and an ID column

The script will output a JSON file with the following structure:
[
    {
        "id_text": int,
        "NEs": [
            {
                "NE": str,
                "id_names": [int]
                "sims": [float],
                "names": [str],
            }
        ]
    }
]

Usage:
    python ner_minimal.py input_names.csv input_texts.csv output.json [--threshold 0.0 --text_str_col text \
    --text_id_col id_text --names_str_col name --names_id_col id_name --sep ";" --sep_names ";" --sep_texts ";" \
    --encoding "utf-8" --ngram_min 2 --ngram_max 2 --min_df 2 --ner_labels "PER,LOC" --analyser "char" \
    --nlp_model "fr_core_news_lg"]

Author: Marceau Hernandez <git@marceau-h.fr>
License: AGPL-3.0
"""

import json
from pathlib import Path

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_THRESHOLD = 0.0
DEFAULT_NLP_MODEL = "fr_core_news_lg"


def extract_ners(text: str, nlp: spacy.Language, ner_labels: str = "PER,LOC") -> list[str]:
    """
    Extracts named entities from a text
    :param text: The text to extract named entities from
    :return: A list of named entities
    """
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ner_labels.split(",")]


def main(
        input_names: str | Path,
        input_texts: str | Path,
        output: str | Path,
        threshold: float = DEFAULT_THRESHOLD,
        text_str_col: str = "text",
        text_id_col: str = "id_text",
        names_str_col: str = "name",
        names_id_col: str = "id_name",
        sep: str = "",
        sep_names: str = "",
        sep_texts: str = "",
        encoding: str = "utf-8",
        ngram_min: int = 2,
        ngram_max: int = 2,
        min_df: int = 2,
        ner_labels: str = "PER,LOC",
        analyser: str = "char",
        nlp_model: str = DEFAULT_NLP_MODEL,
):
    if isinstance(input_names, str):
        input_names = Path(input_names)

    if isinstance(input_texts, str):
        input_texts = Path(input_texts)

    if isinstance(output, str):
        output = Path(output)

    nlp = spacy.load(nlp_model)

    if sep:
        names_df = pd.read_csv(input_names, encoding=encoding, sep=sep).fillna("")
        texts_df = pd.read_csv(input_texts, encoding=encoding, sep=sep).fillna("")
    else:
        names_df = pd.read_csv(
            input_names,
            encoding=encoding,
            **({"sep": sep_names} if sep_names else {}),
        ).fillna("")
        texts_df = pd.read_csv(
            input_texts,
            encoding=encoding,
            **({"sep": sep_texts} if sep_texts else {}),
        ).fillna("")

    docs = names_df[names_str_col].tolist() + texts_df[text_str_col].tolist()

    tfidf = TfidfVectorizer(analyzer=analyser, ngram_range=(ngram_min, ngram_max), min_df=min_df)
    tfidf.fit(docs)

    texts_df["ners"] = texts_df[text_str_col].apply(lambda x: extract_ners(x, nlp, ner_labels))

    names_vectors = tfidf.transform(names_df[names_str_col].tolist())

    mega_struct = []
    for i, row in texts_df.iterrows():
        sims_by_ner = []
        for ner in row["ners"]:
            vect = tfidf.transform([ner])
            sims = cosine_similarity(names_vectors, vect)

            sims_upper_than_tresh = sims > threshold

            sims_upper_indices = sims_upper_than_tresh.nonzero()[0]
            sims_upper_values = sims[sims_upper_indices].flatten()
            sims_upper_in_df = names_df.iloc[sims_upper_indices]

            sims_by_ner.append(
                {
                    "NE": ner,
                    "id_names": sims_upper_in_df[names_id_col].tolist(),
                    "sims": sims_upper_values.tolist(),
                    "names": sims_upper_in_df[names_str_col].tolist(),
                }
            )

        mega_struct.append(
            {
                "id_text": row[text_id_col],
                "NEs": sims_by_ner,
            }
        )

    with output.open("w", encoding=encoding) as f:
        json.dump(mega_struct, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.description = __doc__

    parser.add_argument("input_names", help="Path to the CSV file with names to compare to")
    parser.add_argument("input_texts", help="Path to the CSV file with texts to extract named entities from")
    parser.add_argument("output", help="Path to the output JSON file")

    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold for similarity")
    parser.add_argument("--text_str_col", default="text", help="Name of the column with texts")
    parser.add_argument("--text_id_col", default="id_text", help="Name of the column with text IDs")
    parser.add_argument("--names_str_col", default="name", help="Name of the column with names")
    parser.add_argument("--names_id_col", default="id_name", help="Name of the column with name IDs")
    parser.add_argument("--sep", default="", help="Separator for both CSV files")
    parser.add_argument("--sep_names", default="", help="Separator for the names CSV file")
    parser.add_argument("--sep_texts", default="", help="Separator for the texts CSV file")
    parser.add_argument("--encoding", default="utf-8", help="Encoding for the CSV files")
    parser.add_argument("--ngram_min", type=int, default=2, help="Minimum n-gram size")
    parser.add_argument("--ngram_max", type=int, default=2, help="Maximum n-gram size")
    parser.add_argument("--min_df", type=int, default=2, help="Minimum document frequency")
    parser.add_argument("--ner_labels", default="PER,LOC", help="Named entity labels to extract")
    parser.add_argument("--analyser", default="char", help="Analyser for the TfidfVectorizer")
    parser.add_argument("--nlp_model", default=DEFAULT_NLP_MODEL, help="Spacy NLP model")

    args = parser.parse_args()

    main(**vars(args))


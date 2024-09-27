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
### Importing libraries
import gc
import json
from pathlib import Path

import numpy as np
import spacy
import pandas as pd
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

### Constants (can be overridden by CLI arguments)
DEFAULT_THRESHOLD = 0.0 # Default threshold for similarity, if the similarity is above this value, the name is considered a match
DEFAULT_NLP_MODEL = "fr_core_news_lg" # Default Spacy NLP model, the one used for named entity recognition
TEMP_DIR = Path("tempdir") # Temporary directory to store the tfidf matrix and the similarity matrix
TEMP_DIR.mkdir(exist_ok=True, parents=True) # Make the dir if non-existent

### Functions definition
def extract_ners(text: str, nlp: spacy.Language, ner_labels: str = "PER,LOC") -> list[str]:
    """
    Extracts named entities from a text using a Spacy NLP model
    :param text: The text to extract named entities from
    :param nlp: The Spacy NLP model to use for named entity recognition
    :param ner_labels: A comma-separated list of named entity labels to extract (str)
    :return: A list of named entities found in the text that match the specified ner_labels
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
) -> None:
    """
    Main function to extract named entities from texts and compare them to a list of names
    :param input_names: Path to the CSV file with names to compare to
    :param input_texts: Path to the CSV file with texts to extract named entities from
    :param output: Path to the output JSON file
    :param threshold: Threshold for similarity (if the similarity is above this value, the name is considered a match)
    :param text_str_col: Name of the column with texts in the texts CSV file
    :param text_id_col: Name of the column with text IDs in the texts CSV file
    :param names_str_col: Name of the column with names in the names CSV file
    :param names_id_col: Name of the column with name IDs in the names CSV file
    :param sep: Separator for both CSV files
    :param sep_names: Separator for the names CSV file
    :param sep_texts: Separator for the texts CSV file
    :param encoding: Encoding for the CSV files
    :param ngram_min: Minimum n-gram size for the TfidfVectorizer
    :param ngram_max: Maximum n-gram size for the TfidfVectorizer
    :param min_df: Minimum document frequency for the TfidfVectorizer (2 by default, if a word appears in less than 2 documents, it is ignored)
    :param ner_labels: Named entity labels to extract (comma-separated list)
    :param analyser: Analyser for the TfidfVectorizer (char by default, can be "word" or "char" or "char_wb" or even a custom analyser)
    :param nlp_model: Spacy NLP model to use for named entity recognition
    :return: None, as the results are saved in the output JSON file
    """
    # The input_names, input_texts and output arguments can be either strings or Path objects
    # If they are strings, they are converted to Path objects
    # Path objects are used to handle file paths in a platform-independent way and check if the file exists or not
    if isinstance(input_names, str): # If the input_names is a string, convert it to a Path object (if it's already a Path object, do nothing)
        input_names = Path(input_names) # Convert the string to a Path object

    if isinstance(input_texts, str): # Same as above, but for the input_texts
        input_texts = Path(input_texts) # Convert the string to a Path object

    if isinstance(output, str): # Same as above, but for the output
        output = Path(output) # Convert the string to a Path object

    assert input_names.exists(), f"File not found: {input_names}" # Check if the input_names file exists
    assert input_texts.exists(), f"File not found: {input_texts}" # Check if the input_texts file exists
    assert not output.is_dir(), f"Output path is a directory: {output}" # Check if the output path is not a directory

    nlp = spacy.load(nlp_model) # Load the Spacy NLP model

    if sep: # If the sep argument is not empty, use it as the separator for both CSV files
        names_df = pd.read_csv(input_names, encoding=encoding, sep=sep).fillna("")
        texts_df = pd.read_csv(input_texts, encoding=encoding, sep=sep).fillna("")
    else: # If the sep argument is empty, use sep_names and sep_texts as the separators for the names and texts CSV files respectively
        names_df = pd.read_csv(
            input_names,
            encoding=encoding,
            **({"sep": sep_names} if sep_names else {}), # If sep_names is not empty, use it as the separator for the names CSV file
        ).fillna("") # Fill NaN (empty values) with an empty string
        texts_df = pd.read_csv(
            input_texts,
            encoding=encoding,
            **({"sep": sep_texts} if sep_texts else {}), # If sep_texts is not empty, use it as the separator for the texts CSV file
        ).fillna("") # Same as above, fill NaN values with an empty string

    # Combine the names and texts columns in a single list to fit the TfidfVectorizer
    docs = names_df[names_str_col].tolist() + texts_df[text_str_col].tolist()

    # Instantiate the TfidfVectorizer with the specified parameters and fit it to the documents list (to have the vocabulary
    # and the IDF weights)
    tfidf = TfidfVectorizer(analyzer=analyser, ngram_range=(ngram_min, ngram_max), min_df=min_df)
    tfidf.fit(docs)

    # Extract named entities from the texts using the Spacy NLP model and make a new column in the texts DataFrame
    texts_df["ners"] = texts_df[text_str_col].apply(lambda x: extract_ners(x, nlp, ner_labels))

    # Transform the names into vectors using the TfidfVectorizer (from the names DataFrame)
    names_vectors = tfidf.transform(names_df[names_str_col])

    names_vectors_shape = names_vectors.shape
    temp_tfidf_file = TEMP_DIR / "temp_tfidf.npy"
    fp = np.memmap(temp_tfidf_file, dtype='float32', mode='w+', shape=names_vectors_shape)
    fp[:] = names_vectors[:]
    del fp
    del names_vectors
    gc.collect()

    names_vectors = np.memmap(temp_tfidf_file, dtype='float32', mode='r', shape=names_vectors_shape)

    # Iterate over the texts DataFrame and compare the named entities to the names using cosine similarity
    # If the similarity is above the threshold, add the name to the results
    # The results are saved in a list of dictionaries (`mega_struct`) with the following structure:
    # [
    #     {
    #         "id_text": int,
    #         "NEs": [
    #             {
    #                 "NE": str,
    #                 "id_names": [int],
    #                 "sims": [float],
    #                 "names": [str],
    #             }
    #         ]
    #     }
    # ]
    mega_struct = [] # Initialize the mega_struct list
    for i, row in tqdm(texts_df.iterrows(), total=len(texts_df), desc="Processing texts"): # Iterate over the texts DataFrame
        sims_by_ner = [] # Initialize the sims_by_ner list
        for ner in row["ners"]: # Iterate over the named entities found in the text
            vect = tfidf.transform([ner]) # Transform the named entity into a vector using the TfidfVectorizer
            sims = cosine_similarity(names_vectors, vect) # Compute the cosine similarity between newly transformed vector and the names vectors

            sims_shape = sims.shape
            temp_sim_file ="temp_sim.npy"
            fp = np.memmap(temp_sim_file, dtype='float32', mode='w+', shape=sims_shape)
            fp[:] = sims[:]
            del fp
            del sims
            gc.collect()

            sims = np.memmap(temp_sim_file, dtype='float32', mode='r', shape=sims_shape)

            sims_upper_than_tresh = sims > threshold # Check if the similarity is above the threshold

            sims_upper_indices = sims_upper_than_tresh.nonzero()[0] # Get the indices of the names with a similarity above the threshold
            sims_upper_values = sims[sims_upper_indices].flatten() # Get the similarity values
            sims_upper_in_df = names_df.iloc[sims_upper_indices] # Get the names DataFrame rows with a similarity above the threshold

            # Add the results to the sims_by_ner list, for each named entity found in the text
            # The results are saved in a dictionary with the following structure:
            # {
            #     "NE": str,
            #     "id_names": [int],
            #     "sims": [float],
            #     "names": [str],
            # }
            sims_by_ner.append(
                {
                    "NE": ner,
                    "id_names": sims_upper_in_df[names_id_col].tolist(),
                    "sims": sims_upper_values.tolist(),
                    "names": sims_upper_in_df[names_str_col].tolist(),
                }
            )

            del sims
            del sims_upper_than_tresh
            del sims_upper_indices
            del sims_upper_values
            del sims_upper_in_df
            gc.collect()

        # Add the results to the mega_struct list, for each text
        # The results are saved in a dictionary with the following structure:
        # {
        #     "id_text": int,
        #     "NEs": [sims_by_ner]
        # }
        mega_struct.append(
            {
                "id_text": row[text_id_col],
                "NEs": sims_by_ner,
            }
        )

    # Save the results in the output JSON file
    with output.open("w", encoding=encoding) as f:
        json.dump(mega_struct, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # If used as a cli, import the ArgumentParser class from the argparse module (to parse command-line arguments)
    from argparse import ArgumentParser

    # Create a new ArgumentParser object with a description of the script (the docstring at the beginning of the script)
    parser = ArgumentParser()
    parser.description = __doc__

    # Add the required and positional arguments to the ArgumentParser object
    parser.add_argument("input_names", help="Path to the CSV file with names to compare to")
    parser.add_argument("input_texts", help="Path to the CSV file with texts to extract named entities from")
    parser.add_argument("output", help="Path to the output JSON file")

    # Add optional arguments to the ArgumentParser object, to override the default values of the constants and
    # the main function parameters
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

    # Parse the command-line arguments and store them in the args variable
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(**vars(args))

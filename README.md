# NERresistance
## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)

## Introduction
This repo contains a script to resolve an issue that I have been confronted to.
The script finds named entities using the spaCy library and then matches them to a list of names in a CSV file.
The script then returns the named entities that are in the list with their corresponding similarity score.

## Installation
Clone this repository and install the required packages (preferably in a virtual environment).
This is the recommended way to install the script.

1. Clone the repository
```bash
git clone https://github.com/Marceau-h/NERresistance.git
cd NERresistance
```

2. Create a virtual environment and activate it
```bash
python3.12 -m venv venv
source venv/bin/activate
```

3. Install the required packages
```bash
pip install -r requirements.txt
```

## Usage
The script takes two CSV files as input: one containing the list of names and the other containing the texts. They can be the same file. The script then returns a JSON file containing the named entities that are in the list with their corresponding similarity score.

This is how you can use the script:

0. Activate the virtual environment
```bash
source venv/bin/activate
```

1. Run the script
```bash
python ner_minimal.py input_names.csv input_texts.csv output.json [--threshold 0.0 --text_str_col text \
    --text_id_col id_text --names_str_col name --names_id_col id_name --sep ";" --sep_names ";" --sep_texts ";" \
    --encoding "utf-8" --ngram_min 2 --ngram_max 2 --min_df 2 --ner_labels "PER,LOC" --analyser "char" \
    --nlp_model "fr_core_news_lg"]
```

The script takes the following arguments:
- `input_names.csv`: the path to the CSV file containing the list of names
- `input_texts.csv`: the path to the CSV file containing the texts
- `output.json`: the path to the output JSON file

- `--threshold`: the threshold for the similarity score
- `--text_str_col`: the name of the column containing the texts in the input_texts.csv file
- `--text_id_col`: the name of the column containing the IDs of the texts in the input_texts.csv file
- `--names_str_col`: the name of the column containing the names in the input_names.csv file
- `--names_id_col`: the name of the column containing the IDs of the names in the input_names.csv file
- `--sep`: the separator used in the input_texts.csv file
- `--sep_names`: the separator used in the input_names.csv file
- `--sep_texts`: the separator used in the input_texts.csv file
- `--encoding`: the encoding of the input CSV files
- `--ngram_min`: the minimum size of the n-grams
- `--ngram_max`: the maximum size of the n-grams
- `--min_df`: the minimum frequency of the n-grams
- `--ner_labels`: the labels of the named entities to extract
- `--analyser`: the type of analyser to use for the vectorizer
- `--nlp_model`: the spaCy model to use
- `--echantillon_texts`: the number of texts to sample
- `--echantillon_names`: the number of names to sample

## Contributing
If you want to contribute to this project, you can open an issue or a pull request, and I will be happy to discuss it with you. You can also contact me at [git@marceau-h.fr](mailto:git@marceau-h.fr).

## License
This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

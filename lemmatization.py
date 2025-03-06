import spacy

# Charger les modèles SpaCy pour l'anglais et le français
nlp_en = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp_fr = spacy.load("fr_core_news_sm", disable=["ner", "parser"])

def lemmatize_file(input_file, output_file, nlp_model):
    """Applique la lemmatisation sur un fichier texte ligne par ligne."""
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Lemmatisation ligne par ligne
    lemmatized_lines = [" ".join([token.lemma_ for token in nlp_model(line)]) for line in lines]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lemmatized_lines))

# Lemmatisation des corpus Europarl et EMEA
for corpus in ["Europarl_train_100k", "Europarl_dev_3750", "Europarl_test_500",
               "Emea_train_10k", "Emea_test_500"]:
    lemmatize_file(f"data/split/{corpus}.en", f"data/split/{corpus}.lemma.en", nlp_en)
    lemmatize_file(f"data/split/{corpus}.fr", f"data/split/{corpus}.lemma.fr", nlp_fr)

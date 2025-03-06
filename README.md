## **Description du projet**
Ce projet a été entièrement réalisé sur **Google Colab**, permettant une exécution **directe** des commandes **sans installation manuelle**.  
Il vise à entraîner et évaluer un **modèle de traduction neuronale** basé sur **OpenNMT** en utilisant des corpus bilingues anglais-français.  

Nous explorons **deux approches linguistiques** :
1. **Modèle en forme fléchie** (mots conjugués et accordés)
2. **Modèle en forme lemmatisée** (mots réduits à leur racine lexicale)

L'objectif est d'analyser **l'impact de la lemmatisation** sur la qualité des traductions en utilisant le **score BLEU**.

 **Exécution facile** : Copiez-collez chaque **cellule** de code dans un notebook **Google Colab** et exécutez-la directement.  
 **Attention** : L'entraînement du modèle peut être long.  

### Étape 1 : Installation des Dépendances
 **À exécuter directement sur Colab**

```bash
!pip install torch torchvision torchaudio
!pip install OpenNMT-py sacrebleu stanza spacy

# Clone du dépôt Moses pour la normalisation
!git clone https://github.com/moses-smt/mosesdecoder.git

# Téléchargement des modèles linguistiques
import stanza
stanza.download("en")  # Anglais
stanza.download("fr")  # Français

import spacy
spacy.cli.download("en_core_web_sm")
spacy.cli.download("fr_core_news_sm")
```

---

###  Étape 2 : Vérification d'OpenNMT sur un Petit Corpus
**Test rapide sur un petit corpus anglais-allemand**

```bash
!wget https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
!tar xf toy-ende.tar.gz
!cd toy-ende
!onmt_build_vocab -config toy_en_de.yaml -save_data toy_data
!onmt_train -config toy_en_de.yaml
!onmt_translate -model toy_model.pt -src toy_data/src-test.txt -output toy_data/pred.txt
!sacrebleu toy_data/tgt-test.txt -i toy_data/pred.txt -m bleu -b
```

---

###  Étape 3 : Préparation des Données
**Téléchargement des corpus Europarl et EMEA et séparation des données**

```bash
!mkdir -p data/Europarl data/EMEA
!wget -O data/Europarl/en-fr.txt.zip https://object.pouta.csc.fi/OPUS-Europarl/v8/moses/en-fr.txt.zip
!unzip data/Europarl/en-fr.txt.zip -d data/Europarl

!wget -O data/EMEA/en-fr.txt.zip https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/en-fr.txt.zip
!unzip data/EMEA/en-fr.txt.zip -d data/EMEA
```

 **Séparation des données en ensembles d'entraînement, validation et test**

```bash
!mkdir -p data/split

# Europarl
!head -100000 data/Europarl/Europarl.en-fr.en > data/split/Europarl_train_100k.en
!head -100000 data/Europarl/Europarl.en-fr.fr > data/split/Europarl_train_100k.fr

!tail -n +100001 data/Europarl/Europarl.en-fr.en | head -3750 > data/split/Europarl_dev_3750.en
!tail -n +100001 data/Europarl/Europarl.en-fr.fr | head -3750 > data/split/Europarl_dev_3750.fr

!tail -n +103751 data/Europarl/Europarl.en-fr.en | head -500 > data/split/Europarl_test_500.en
!tail -n +103751 data/Europarl/Europarl.en-fr.fr | head -500 > data/split/Europarl_test_500.fr

# EMEA
!head -10000 data/EMEA/EMEA.en-fr.en > data/split/Emea_train_10k.en
!head -10000 data/EMEA/EMEA.en-fr.fr > data/split/Emea_train_10k.fr

!tail -n +13751 data/EMEA/EMEA.en-fr.en | head -500 > data/split/Emea_test_500.en
!tail -n +13751 data/EMEA/EMEA.en-fr.fr | head -500 > data/split/Emea_test_500.fr
```

###  Étape 4 : Entraînement du Modèle (Forme Fléchie)

---

### **1Fichier de Configuration OpenNMT**

```yaml
%%writefile /content/opennmt_config.yaml
##  Données d'entraînement et validation
data:
    corpus_1:
        path_src: /content/data/split/Europarl_train_100k.tok.true.clean.en
        path_tgt: /content/data/split/Europarl_train_100k.tok.true.clean.fr
        transforms: [filtertoolong]
        weight: 1
    valid:
        path_src: /content/data/split/Europarl_dev_3750.tok.true.en
        path_tgt: /content/data/split/Europarl_dev_3750.tok.true.fr
        transforms: [filtertoolong]

##  Vocabulaire
src_vocab: /content/data/opennmt/vocab.src
tgt_vocab: /content/data/opennmt/vocab.tgt

##  Modèle Transformer (Optimisé pour le GPU A100 utilisable grâce à Colab Pro)
model_task: seq2seq
model_type: text
encoder_type: transformer
decoder_type: transformer
layers: 6
heads: 8
hidden_size: 512
word_vec_size: 512
transformer_ff: 2048
dropout: 0.1

## Optimisation et accélération
optim: adam
learning_rate: 0.0005  #  Réduction pour éviter instabilité
warmup_steps: 4000
accum_count: 8  #  Accumulation pour simuler batch plus grand

##  Gestion efficace du batch et de la mémoire GPU
batch_size: 1024  #  Réduction pour éviter OOM
valid_batch_size: 8
max_generator_batches: 2  #  Économie mémoire pour l'inférence
batch_type: "tokens"  #  Ajustement dynamique pour éviter les pics mémoire

##  Entraînement et checkpoints
train_steps: 10000  #  Éviter surcharge mémoire sur long training
valid_steps: 1000
save_checkpoint_steps: 2000
save_model: /content/model_opennmt/model

##  GPU et performance (Optimisé)
gpu_ranks: [0]
world_size: 1
precision: "float16"  #  FP16 pour économiser mémoire
num_threads: 4  #  Chargement plus fluide
queue_size: 10000  #  Équilibrage entre vitesse et mémoire

##  Gestion avancée de la mémoire GPU
save_data: /content/model_opennmt
overwrite: True
disable_mem_redundancy: True  #  Réduit la redondance mémoire
exp_global_attn: True  #  Optimisation de l'attention pour mémoire limitée
```

---

### **2️ Création du Vocabulaire**
 **Générer le vocabulaire utilisé pour l'entraînement.**

```bash
!onmt_build_vocab -config /content/opennmt_config.yaml -save_data /content/data/opennmt -n_sample 10000
```

---

### **3️ Entraînement du Modèle**
 **Lancer l'entraînement du modèle Transformer.**

```bash
!onmt_train -config /content/opennmt_config.yaml
```

---

### **4️ Traduction**
 **Utilisation du modèle entraîné pour traduire les phrases de test.**

```bash
!onmt_translate -model /content/model_opennmt/model_step_10000.pt \
                -src /content/data/split/Europarl_test_500.tok.true.en \
                -output /content/data/split/Europarl_test_500_translated.fr \
                -gpu 0
```

---

### **5️ Évaluation avec SacreBLEU**
 **Calcul de la qualité de la traduction à l'aide du score BLEU.**

```bash
!sacrebleu /content/data/split/Europarl_test_500.tok.true.fr \
           -i /content/data/split/Europarl_test_500_translated.fr \
           -m bleu -b --force
```

###  Étape 5 : Entraînement du Modèle (Forme Lemmatisée)

---

### **1️ Fichier de Configuration OpenNMT**

```yaml
%%writefile /content/opennmt_config.yaml
## Configuration pour corpus lemmatisé
data:
    corpus_1:
        path_src: /content/data/split/Europarl_train_100k.lemma.en
        path_tgt: /content/data/split/Europarl_train_100k.lemma.fr
        transforms: [filtertoolong]
        weight: 1
    corpus_2:
        path_src: /content/data/split/Emea_train_10k.lemma.en
        path_tgt: /content/data/split/Emea_train_10k.lemma.fr
        transforms: [filtertoolong]
    valid:
        path_src: /content/data/split/Europarl_dev_3750.lemma.en
        path_tgt: /content/data/split/Europarl_dev_3750.lemma.fr
        transforms: [filtertoolong]

##  Paramètres inchangés
src_vocab: /content/data/opennmt/vocab.src
tgt_vocab: /content/data/opennmt/vocab.tgt
train_steps: 10000
save_checkpoint_steps: 2000
save_model: /content/model_opennmt/model_lemma
gpu_ranks: [0]
world_size: 1
precision: "float16"
batch_size: 1024
```

---

### **2️ Fichier de Lemmatisation**
 **Ce script applique la lemmatisation sur tous les corpus avant l'entraînement.**

```python
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
```

Exécutez ensuite la lemmatisation :

```bash
!python lemmatization.py
```

---

### **3️ Création du Vocabulaire**
 **Générer le vocabulaire utilisé pour l'entraînement.**

```bash
!onmt_build_vocab -config /content/opennmt_config.yaml -save_data /content/data/opennmt -n_sample 10000
```

---

### **4️ Entraînement du Modèle**
 **Lancer l'entraînement du modèle Transformer sur Colab avec les données lemmatisées.**

```bash
!onmt_train -config /content/opennmt_config.yaml
```

---

### **5️ Traduction**
 **Utilisation du modèle entraîné pour traduire les phrases de test (Europarl et EMEA).**

####  **Traduction Europarl**
```bash
!onmt_translate -model /content/model_opennmt/model_lemma_step_10000.pt \
                -src /content/data/split/Europarl_test_500.lemma.en \
                -output /content/data/split/Europarl_test_500_translated_lemma.fr \
                -gpu 0
```

#### 🔹 **Traduction EMEA**
```bash
!onmt_translate -model /content/model_opennmt/model_lemma_step_10000.pt \
                -src /content/data/split/Emea_test_500.lemma.en \
                -output /content/data/split/Emea_test_500_translated_lemma.fr \
                -gpu 0
```

---

### **6️ Évaluation avec SacreBLEU**
 **Calcul de la qualité de la traduction pour Europarl et EMEA.**

####  **Évaluation Europarl**
```bash
!sacrebleu /content/data/split/Europarl_test_500.lemma.fr \
           -i /content/data/split/Europarl_test_500_translated_lemma.fr \
           -m bleu -b --force
```

####  **Évaluation EMEA**
```bash
!sacrebleu /content/data/split/Emea_test_500.lemma.fr \
           -i /content/data/split/Emea_test_500_translated_lemma.fr \
           -m bleu -b --force
```

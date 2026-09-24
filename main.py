
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import spacy
from nltk.stem import PorterStemmer
from transformers import AutoTokenizer

app = FastAPI(title="NLP Analysis Platform")

nlp = spacy.load("en_core_web_sm")

stemmer = PorterStemmer()

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/analyze")
def analyze(data: TextInput):

    text = data.text

    if not text.strip():
        return {
            "error": "Please enter some text."
        }

    doc = nlp(text)

    words = [
        token.text
        for token in doc
        if not token.is_space
    ]

    stopwords = [
        token.text
        for token in doc
        if token.is_stop
    ]

    stopwords_removed = [
        token.text
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
    ]

    punctuation = [
        token.text
        for token in doc
        if token.is_punct
    ]

    lemmas = []

    for token in doc:

        if not token.is_space:

            lemmas.append({
                "word": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_
            })

    stemming = []

    for token in doc:

        if token.is_alpha:

            stemming.append({
                "word": token.text,
                "stem": stemmer.stem(token.text)
            })

    pos_tags = []

    for token in doc:

        if not token.is_space:

            pos_tags.append({
                "word": token.text,
                "pos": token.pos_,
                "detailed_tag": token.tag_,
                "lemma": token.lemma_
            })

    pos_counts = {}

    for token in doc:

        if not token.is_space:

            pos = token.pos_

            if pos in pos_counts:
                pos_counts[pos] += 1
            else:
                pos_counts[pos] = 1

    pos_distribution = [
        {
            "pos": pos,
            "frequency": frequency
        }
        for pos, frequency in pos_counts.items()
    ]

    morphology = []

    for token in doc:

        if not token.is_space:

            morphology.append({
                "word": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "morphology": str(token.morph)
            })

    dependencies = []

    for token in doc:

        if not token.is_space:

            dependencies.append({
                "word": token.text,
                "pos": token.pos_,
                "dependency": token.dep_,
                "head": token.head.text
            })

    entities = []

    for ent in doc.ents:

        entities.append({
            "entity": ent.text,
            "type": ent.label_,
            "description": spacy.explain(ent.label_)
        })

    transformer_tokens = tokenizer.tokenize(text)

    encoded = tokenizer(
        text,
        return_attention_mask=True
    )

    token_ids = encoded["input_ids"]

    attention_mask = encoded["attention_mask"]

    return {
        "words": words,
        "word_count": len(words),
        "stopwords": stopwords,
        "stopwords_removed": stopwords_removed,
        "punctuation": punctuation,
        "lemmas": lemmas,
        "stemming": stemming,
        "pos_tags": pos_tags,
        "pos_distribution": pos_distribution,
        "morphology": morphology,
        "dependencies": dependencies,
        "entities": entities,
        "transformer_tokens": transformer_tokens,
        "token_ids": token_ids,
        "attention_mask": attention_mask
    }


import pandas as pd
import spacy
import logging

logger = logging.getLogger(__name__)

_NLP = None

def _get_nlp():
    global _NLP
    if _NLP is None:
        logger.info("Loading spaCy model...")
        _NLP = spacy.load(
            "en_core_web_sm",
            disable=["parser", "ner", "tagger"]  # keep only tokenizer + lemmatizer
        )
        logger.info("spaCy model loaded.")
    return _NLP


def preprocess_text(
    df: pd.DataFrame,
    column_name_to_clean: str = 'transcript',
    new_column_name: str = 'transcript_cleaned',
    batch_size: int = 200,
    n_process: int = -1,  # use all cores
    log_every: int = 1000
) -> pd.DataFrame:

    nlp = _get_nlp()

    texts = df[column_name_to_clean].fillna("").str.lower().tolist()
    total = len(texts)

    logger.info(f"Starting preprocessing of {total} documents...")

    cleaned = []
    count = 0

    docs = nlp.pipe(texts, batch_size=batch_size, n_process=n_process)

    for doc in docs:
        cleaned.append(
            " ".join(
                token.lemma_
                for token in doc
                if not token.is_punct and not token.is_space
            )
        )

        count += 1
        if count % log_every == 0:
            logger.info(f"Processed {count}/{total} documents ({count/total:.2%})")

    logger.info("Preprocessing completed.")

    dfc = df.copy()
    dfc[new_column_name] = cleaned
    return dfc.drop(columns=[column_name_to_clean])



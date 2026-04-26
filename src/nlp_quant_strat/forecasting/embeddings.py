"""
Embedding builders for earnings call transcripts.

Two approaches:
- TFIDFBuilder               : sparse TF-IDF bag-of-words features (fast, interpretable).
                               Best used with preprocessed (stemmed) text — set TEXT_COLUMN
                               to 'transcript_cleaned' in config when available.
- SentenceTransformerBuilder : dense embeddings via chunk + mean-pool (semantic).
                               Best used with raw text. Long transcripts are split into
                               word-level chunks of CHUNK_SIZE words, each encoded
                               independently, then mean-pooled into one document vector.

Both builders return DataFrames with columns
    [filing_date, asset, feat_0, feat_1, ...]
aligned to the unique (filing_date, asset) events in mapping_df.

Caching: pass `aws` to build() to enable S3 caching (load_or_compute in config).

Requires for SentenceTransformerBuilder:
    pip install sentence-transformers
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Tuple, cast

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_quant_strat.utils.config import Config

logger = logging.getLogger(__name__)


def _s3_cache_key(s3_prefix: str, prefix: str, params: dict) -> str:
    """
    Build a deterministic S3 cache key for a set of embedding parameters.

    The key is ``<s3_prefix>/<prefix>_<md5[:10]>.parquet`` where the MD5 is
    computed over the JSON-serialised ``params`` dict (keys sorted for stability).
    This ensures that different parameter combinations produce different keys
    while keeping the key short and human-readable.
    """
    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    return s3_prefix.rstrip("/") + f"/{prefix}_{key}.parquet"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_event_texts(
    mapping_df: pd.DataFrame,
    transcripts_df: pd.DataFrame,
    text_column: str,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns [filing_date, asset, <text_column>],
    one row per unique (filing_date, asset) event.

    Looks for <text_column> in mapping_df first (fast path); if absent,
    joins with transcripts_df via (filing_date, ticker).
    """
    events = (
        mapping_df[["filing_date", "asset"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if text_column in mapping_df.columns:
        text_df = (
            mapping_df[["filing_date", "asset", text_column]]
            .drop_duplicates(subset=["filing_date", "asset"])
            .reset_index(drop=True)
        )
        return events.merge(text_df, on=["filing_date", "asset"], how="left")

    # text_column not in mapping_df — join via transcripts_df
    if "ticker" not in mapping_df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found in mapping_df and mapping_df "
            "has no 'ticker' column for joining with transcripts_df."
        )

    tdf = transcripts_df.copy()
    if not isinstance(tdf.index, pd.RangeIndex):
        tdf = tdf.reset_index()
    if "ticker_api" in tdf.columns:
        tdf = tdf.rename(columns={"ticker_api": "ticker"})
    tdf["filing_date"] = pd.to_datetime(tdf["filing_date"])

    if text_column not in tdf.columns:
        raise ValueError(
            f"Text column '{text_column}' not found in mapping_df or transcripts_df."
        )

    # Deduplicate transcripts_df on (filing_date, ticker) before merging to prevent
    # row expansion when there are multiple transcript entries per event.
    tdf = tdf.drop_duplicates(subset=["filing_date", "ticker"], keep="last")

    merged = (
        mapping_df[["filing_date", "asset", "ticker"]]
        .drop_duplicates(subset=["filing_date", "asset"])
        .reset_index(drop=True)
        .merge(
            tdf[["filing_date", "ticker", text_column]],
            on=["filing_date", "ticker"],
            how="left",
        )
    )
    return merged[["filing_date", "asset", text_column]]


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

class TFIDFBuilder:
    """
    TF-IDF embedding builder.

    Fits a TfidfVectorizer on the full transcript corpus and transforms each
    event's text into a fixed-length dense feature vector.

    Parameters (from config.embeddings_tfidf)
    -----------------------------------------
    max_features : int   — vocabulary size cap (default 200)
    ngram_range  : tuple — (min_n, max_n) for n-grams (default (1, 2))
    text_column  : str   — column name containing the transcript text.
                           Use 'transcript_cleaned' for preprocessed text.
    """

    def __init__(self, config: Config):
        cfg = config.embeddings_tfidf or {}
        self.max_features: int = cfg.get("max_features", 200)
        self.ngram_range: Tuple[int, int] = cast(Tuple[int, int], tuple(cfg.get("ngram_range", [1, 2])))
        self.text_column: str = cfg.get("text_column", "transcript")
        self.load_or_compute: str = config.embeddings_load_or_compute
        self.s3_cache_prefix: str | None = config.embeddings_s3_cache_prefix
        self._vectorizer: TfidfVectorizer | None = None

    def build(
        self,
        mapping_df: pd.DataFrame,
        transcripts_df: pd.DataFrame,
        aws=None,
    ) -> pd.DataFrame:
        """
        Fit-transform TF-IDF on the full corpus.

        Parameters
        ----------
        aws : optional AWS client from DataManager. Required for S3 caching.
        mapping_df: mapping_df from data_manager
        transcripts_df: from data_manager
        Returns
        -------
        pd.DataFrame
            Columns: [filing_date, asset, tfidf_0, ..., tfidf_{max_features-1}]
        """
        cache_params = {
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "text_column": self.text_column,
        }

        s3_key = None
        if aws is not None and self.s3_cache_prefix is not None:
            s3_key = _s3_cache_key(self.s3_cache_prefix, "tfidf", cache_params)
            if self.load_or_compute == "load":
                try:
                    result = aws.s3.load(key=s3_key)
                    logger.info("Loaded TF-IDF embeddings from S3: %s", s3_key)
                    return result
                except Exception:
                    logger.info("No S3 cache found at %s, will compute.", s3_key)

        events = _get_event_texts(mapping_df, transcripts_df, self.text_column)
        texts = events[self.text_column].fillna("").tolist()

        logger.info(
            "Fitting TF-IDF (max_features=%d, ngram_range=%s) on %d documents...",
            self.max_features, self.ngram_range, len(texts),
        )

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )

        t0 = time.time()
        logger.info("Running fit_transform — this can take several minutes for large corpora...")
        matrix = self._vectorizer.fit_transform(texts).toarray()
        logger.info(
            "fit_transform done in %.1f s | vocab size: %d | output: %s",
            time.time() - t0, len(self._vectorizer.vocabulary_), matrix.shape,
        )

        feat_cols = [f"tfidf_{i}" for i in range(matrix.shape[1])]
        result = pd.DataFrame(matrix, columns=feat_cols)
        result.insert(0, "filing_date", events["filing_date"].values)
        result.insert(1, "asset", events["asset"].values)

        if s3_key is not None:
            logger.info("Saving TF-IDF embeddings to S3: %s", s3_key)
            aws.s3.upload(src=result, key=s3_key)
            logger.info("Saved.")

        logger.info("TF-IDF built: shape %s", result.shape)
        return result


# ---------------------------------------------------------------------------
# Sentence Transformer
# ---------------------------------------------------------------------------

class SentenceTransformerBuilder:
    """
    Dense embedding builder using sentence-transformers with chunk + mean-pool
    to handle earnings call transcripts that exceed the model's token limit.

    Each transcript is split into non-overlapping word-level chunks of
    CHUNK_SIZE words. Each chunk is encoded independently, and the resulting
    embeddings are mean-pooled into one document-level vector.

    Parameters (from config.embeddings_sentence_transformer)
    ---------------------------------------------------------
    model_name  : str — HuggingFace model identifier (default 'all-MiniLM-L6-v2')
    chunk_size  : int — words per chunk (default 256 ≈ 300–500 subword tokens)
    batch_size  : int — chunks per encoder call (default 32)
    text_column : str — column name containing the raw transcript text
    """

    def __init__(self, config: Config):
        cfg = config.embeddings_sentence_transformer or {}
        self.model_name: str = cfg.get("model_name", "all-MiniLM-L6-v2")
        self.chunk_size: int = cfg.get("chunk_size", 256)
        self.batch_size: int = cfg.get("batch_size", 32)
        self.text_column: str = cfg.get("text_column", "transcript")
        self.load_or_compute: str = config.embeddings_load_or_compute
        self.s3_cache_prefix: str | None = config.embeddings_s3_cache_prefix
        self._model = None

    def _get_model(self):
        """
        Lazy-load the SentenceTransformer model, detecting GPU availability.

        Sets ``self._device`` ('cuda' | 'cpu') and
        ``self._effective_batch_size`` (batch_size × 8 on GPU, batch_size on CPU).
        Subsequent calls return the already-loaded model without reloading.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Install it with: pip install sentence-transformers"
                )
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = "cuda"
                    logger.info("GPU detected: %s — using CUDA.", torch.cuda.get_device_name(0))
                else:
                    self._device = "cpu"
                    logger.info("No CUDA GPU detected — using CPU.")
            except ImportError:
                self._device = "cpu"
                logger.info("torch not importable — using CPU.")

            # Scale batch_size up on GPU: CUDA can saturate with ~8x more chunks at once
            if self._device == "cuda":
                self._effective_batch_size = self.batch_size * 8
                logger.info("GPU batch_size set to %d (cpu base %d × 8).", self._effective_batch_size, self.batch_size)
            else:
                self._effective_batch_size = self.batch_size

            logger.info("Loading sentence-transformer model '%s' on %s...", self.model_name, self._device)
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    def build(
        self,
        mapping_df: pd.DataFrame,
        transcripts_df: pd.DataFrame,
        aws=None,
    ) -> pd.DataFrame:
        """
        Encode each transcript using chunk + mean-pool.

        Parameters
        ----------
        aws : optional AWS client from DataManager. Required for S3 caching.
        mapping_df: mapping_df from data_manager
        transcripts_df: from data_manager
        Returns
        -------
        pd.DataFrame
            Columns: [filing_date, asset, emb_0, ..., emb_{dim-1}]
        """
        cache_params = {
            "model_name": self.model_name,
            "chunk_size": self.chunk_size,
            "text_column": self.text_column,
        }

        s3_key = None
        if aws is not None and self.s3_cache_prefix is not None:
            s3_key = _s3_cache_key(self.s3_cache_prefix, "st", cache_params)
            if self.load_or_compute == "load":
                try:
                    result = aws.s3.load(key=s3_key)
                    logger.info("Loaded sentence-transformer embeddings from S3: %s", s3_key)
                    return result
                except Exception:
                    logger.info("No S3 cache found at %s, will compute.", s3_key)

        model = self._get_model()
        events = _get_event_texts(mapping_df, transcripts_df, self.text_column)
        texts = events[self.text_column].fillna("").tolist()

        logger.info(
            "Encoding %d transcripts with '%s' (chunk_size=%d words, batch_size=%d)...",
            len(texts), self.model_name, self.chunk_size, self.batch_size,
        )

        embeddings = self._encode_chunked(model, texts)

        dim = embeddings.shape[1]
        feat_cols = [f"emb_{i}" for i in range(dim)]
        result = pd.DataFrame(embeddings, columns=feat_cols)
        result.insert(0, "filing_date", events["filing_date"].values)
        result.insert(1, "asset", events["asset"].values)

        if s3_key is not None:
            logger.info("Saving sentence-transformer embeddings to S3: %s", s3_key)
            aws.s3.upload(src=result, key=s3_key)
            logger.info("Saved.")

        logger.info("Sentence-transformer embeddings built: shape %s", result.shape)
        return result

    def _encode_chunked(self, model, texts: list) -> np.ndarray:
        """
        Split each text into word chunks, encode each chunk, mean-pool.
        Empty texts produce a zero vector of the correct dimension.
        Logs progress every ~5% of documents with elapsed time and ETA.
        """
        dim = model.get_embedding_dimension()
        all_embeddings: list[np.ndarray] = []
        total = len(texts)
        log_every = max(1, total // 20)
        t0 = time.time()
        batch_size = getattr(self, "_effective_batch_size", self.batch_size)

        for i, text in enumerate(texts):
            words = text.split()
            if not words:
                all_embeddings.append(np.zeros(dim))
            else:
                chunks = [
                    " ".join(words[j : j + self.chunk_size])
                    for j in range(0, len(words), self.chunk_size)
                ]
                chunk_embs = model.encode(
                    chunks,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                all_embeddings.append(chunk_embs.mean(axis=0))

            if (i + 1) % log_every == 0 or i == total - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else float("inf")
                eta = (total - i - 1) / rate if rate > 0 else 0.0
                logger.info(
                    "Encoded %d/%d (%.0f%%) | %.1f docs/s | ETA %.0f s",
                    i + 1, total, (i + 1) / total * 100, rate, eta,
                )

        return np.vstack(all_embeddings)
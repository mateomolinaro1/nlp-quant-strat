from __future__ import annotations

import glob
import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

from nlp_quant_strat.data.singleton import Singleton

logger = logging.getLogger(__name__)

TRANSCRIPTS_PATH = Path("data/transcripts")


class TranscriptTypes(str, Enum):
    UNPROCESSED = "unprocessed"
    PREPROCESSED = "preprocessed"


class DataLoader(Singleton):
    """
    Singleton data loader with in-memory caching.

    Example:
        loader = DataLoader()
        df_raw = loader.get_data(TranscriptTypes.UNPROCESSED)
        df_clean = loader.get_data("preprocessed")
    """

    _FILE_PATTERNS = {
        TranscriptTypes.UNPROCESSED: "formatted_transcripts_gzip_chunk_*.pkl",
        TranscriptTypes.PREPROCESSED: "formatted_transcripts_preprocessed_gzip_chunk_*.pkl",
    }

    def __init__(self, key: Optional[str | TranscriptTypes] = None, *args, **kwargs) -> None:
        if not hasattr(self, "_initialized"):
            self._data_cache: dict[TranscriptTypes, pd.DataFrame] = {}
            self._initialized = True

        if key is not None:
            self._ensure_data_loaded(key)

    def _normalize_key(self, key: str | TranscriptTypes) -> TranscriptTypes:
        """
        Normalize user input into a TranscriptTypes enum.

        Supports both:
        - TranscriptTypes.UNPROCESSED / PREPROCESSED
        - "unprocessed" / "preprocessed"

        Also supports a few legacy aliases for backward compatibility.
        """
        if isinstance(key, TranscriptTypes):
            return key

        normalized = key.strip().lower()

        legacy_aliases = {
            "formated_transcript_": TranscriptTypes.UNPROCESSED,
            "formatted_transcript_": TranscriptTypes.UNPROCESSED,
            "formated_transcript_preprocessed_": TranscriptTypes.PREPROCESSED,
            "formatted_transcript_preprocessed_": TranscriptTypes.PREPROCESSED,
            "raw": TranscriptTypes.UNPROCESSED,
            "clean": TranscriptTypes.PREPROCESSED,
        }

        if normalized in legacy_aliases:
            return legacy_aliases[normalized]

        try:
            return TranscriptTypes(normalized)
        except ValueError as exc:
            valid_values = [t.value for t in TranscriptTypes]
            raise ValueError(
                f"Unknown key: {key!r}. Expected one of {valid_values} "
                f"or a supported legacy alias."
            ) from exc

    def _ensure_data_loaded(self, key: str | TranscriptTypes) -> None:
        """
        Ensure data for the given key is loaded into cache.
        """
        normalized_key = self._normalize_key(key)
        if normalized_key not in self._data_cache:
            self._data_cache[normalized_key] = self._load_data(normalized_key)

    def get_data(self, key: str | TranscriptTypes) -> pd.DataFrame:
        """
        Return the requested dataset, loading and caching it if needed.
        """
        normalized_key = self._normalize_key(key)
        self._ensure_data_loaded(normalized_key)
        return self._data_cache[normalized_key]

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """
        Backward-compatible property.

        Returns the first cached dataset, or None if cache is empty.
        """
        if not self._data_cache:
            return None
        return next(iter(self._data_cache.values()))

    def clear_cache(self) -> None:
        """
        Clear all cached datasets.
        """
        self._data_cache.clear()

    def clear_cache_for(self, key: str | TranscriptTypes) -> None:
        """
        Clear a specific cached dataset.
        """
        normalized_key = self._normalize_key(key)
        self._data_cache.pop(normalized_key, None)

    def get_cached_keys(self) -> list[str]:
        """
        Return the list of currently cached dataset keys.
        """
        return [key.value for key in self._data_cache.keys()]

    def _load_data(self, key: TranscriptTypes) -> pd.DataFrame:
        """
        Load transcript data based on the provided key.

        Handles chunked pickle files and concatenates them into one DataFrame.
        """
        if key not in self._FILE_PATTERNS:
            raise ValueError(f"Unsupported transcript type: {key}")

        file_pattern = self._FILE_PATTERNS[key]
        chunk_pattern = str(TRANSCRIPTS_PATH / file_pattern)
        chunk_files = sorted(glob.glob(chunk_pattern))

        if not chunk_files:
            raise FileNotFoundError(
                f"No chunk files found for key={key.value!r} "
                f"with pattern: {chunk_pattern}"
            )

        dataframes: list[pd.DataFrame] = []

        for chunk_file in chunk_files:
            try:
                df_chunk = pd.read_pickle(chunk_file)
                dataframes.append(df_chunk)
            except Exception as exc:
                logger.warning("Failed to load chunk %s: %s", chunk_file, exc)

        if not dataframes:
            raise RuntimeError(
                f"No chunks could be loaded successfully for key={key.value!r}"
            )

        combined_df = pd.concat(dataframes, axis=0).sort_index()
        return combined_df
"""
FeatureSet: assembles the ML-ready event-level feature matrix (X, y).

Role in the pipeline
--------------------
FeatureEngineering produces sentiment signals in wide panel format (dates × assets).
EmbeddingBuilders produce embeddings in event format (one row per earnings event).
FeatureSet bridges both worlds:
  1. Looks up each event's sentiment values from the panel DataFrames.
  2. Left-joins the chosen embeddings (TF-IDF, sentence-transformer, or both).
  3. Drops events where any feature is NaN and logs how many were removed.
  4. Optionally applies cross-sectional z-score normalization within each
     filing_date to avoid forward-looking bias.
  5. Returns (X, y) as aligned DataFrames ready for walk-forward CV.

FeatureMode (config-driven via FEATURE_SET.MODE)
------------------------------------------------
  "tfidf"          — TF-IDF embeddings only
  "st"             — sentence-transformer embeddings only
  "sentiment"      — traditional sentiment features only
  "tfidf_sentiment"— TF-IDF + sentiment
  "st_sentiment"   — sentence-transformer + sentiment
  "all"            — TF-IDF + sentence-transformer + sentiment

Normalization
-------------
Not applied here. Scaling (StandardScaler / RobustScaler) is delegated to the
walk-forward CV pipeline so it is fit on the training split only and applied
consistently across models.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from nlp_quant_strat.data.feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)

_TARGET_COLS_PREFIX = "idio_"
_SENTINEL_COLS = {"filing_date", "asset"}


class FeatureMode(str, Enum):
    TFIDF = "tfidf"
    ST = "st"
    SENTIMENT = "sentiment"
    TFIDF_SENTIMENT = "tfidf_sentiment"
    ST_SENTIMENT = "st_sentiment"
    ALL = "all"


class FeatureSet:
    """
    Assembles (X, y) from targets, embeddings, and sentiment features.

    Parameters
    ----------
    mode : str | FeatureMode
        The feature group(s) to include.  Pass one of the FeatureMode values
        (or its string equivalent).  To iterate over multiple modes, instantiate
        FeatureSet once per mode in the outer loop (see EarningsWalkForwardCV
        usage in the debug / pipeline scripts).
    """

    _SENTIMENT_ATTRS = [
        "positive_count", "negative_count", "word_count",
        "polarity", "sentiment_density", "pos_polarity_count_q", "polarity_delta",
    ]

    def __init__(self, mode: str | FeatureMode):
        self.mode = FeatureMode(mode) if isinstance(mode, str) else mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        targets: pd.DataFrame,
        feature_eng: FeatureEngineering | None = None,
        tfidf_embeddings: pd.DataFrame | None = None,
        st_embeddings: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build the ML feature matrix and target matrix.

        Parameters
        ----------
        targets : pd.DataFrame
            Output of TargetBuilder.build(). Columns: [filing_date, asset, idio_Xd, ...]
        feature_eng : FeatureEngineering, optional
            Required when mode includes "sentiment".
        tfidf_embeddings : pd.DataFrame, optional
            Required when mode includes "tfidf". Columns: [filing_date, asset, tfidf_0, ...]
        st_embeddings : pd.DataFrame, optional
            Required when mode includes "st". Columns: [filing_date, asset, emb_0, ...]

        Returns
        -------
        X : pd.DataFrame
            Columns: [filing_date, asset, <feature_cols>]. No NaN rows.
        y : pd.DataFrame
            Columns: [filing_date, asset, idio_1d, idio_5d, idio_21d, idio_63d].
            Row-aligned with X.
        """
        self._validate_inputs(feature_eng, tfidf_embeddings, st_embeddings)

        target_cols = [c for c in targets.columns if c.startswith(_TARGET_COLS_PREFIX)]
        events = targets[["filing_date", "asset"]].copy().reset_index(drop=True)

        feature_parts: list[pd.DataFrame] = []

        # --- Embeddings ---
        if self.mode in (FeatureMode.TFIDF, FeatureMode.TFIDF_SENTIMENT, FeatureMode.ALL):
            feature_parts.append(self._merge_embeddings(events, tfidf_embeddings, "tfidf"))

        if self.mode in (FeatureMode.ST, FeatureMode.ST_SENTIMENT, FeatureMode.ALL):
            feature_parts.append(self._merge_embeddings(events, st_embeddings, "st"))

        # --- Sentiment ---
        if self.mode in (
            FeatureMode.SENTIMENT, FeatureMode.TFIDF_SENTIMENT,
            FeatureMode.ST_SENTIMENT, FeatureMode.ALL,
        ):
            feature_parts.append(self._lookup_sentiment(events, feature_eng))

        # --- Assemble X ---
        X = events.copy()
        for part in feature_parts:
            feat_cols = [c for c in part.columns if c not in _SENTINEL_COLS]
            X = X.join(part[feat_cols])

        # --- Drop NaN feature rows ---
        feat_cols = [c for c in X.columns if c not in _SENTINEL_COLS]
        nan_mask = X[feat_cols].isna().any(axis=1)
        n_dropped = nan_mask.sum()
        if n_dropped > 0:
            logger.warning(
                "Dropping %d/%d events with NaN features (%.1f%%). "
                "Likely cause: embedding join miss or sentiment lookup miss.",
                n_dropped, len(X), n_dropped / len(X) * 100,
            )
        X = X[~nan_mask].reset_index(drop=True)

        # --- Align y to surviving X rows ---
        y = targets[["filing_date", "asset"] + target_cols].merge(
            X[["filing_date", "asset"]], on=["filing_date", "asset"], how="inner"
        ).reset_index(drop=True)

        logger.info(
            "FeatureSet built | mode=%s | X: %s | y: %s",
            self.mode.value, X.shape, y.shape,
        )
        return X, y

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self, feature_eng, tfidf_embeddings, st_embeddings) -> None:
        needs_tfidf = self.mode in (FeatureMode.TFIDF, FeatureMode.TFIDF_SENTIMENT, FeatureMode.ALL)
        needs_st = self.mode in (FeatureMode.ST, FeatureMode.ST_SENTIMENT, FeatureMode.ALL)
        needs_sent = self.mode in (
            FeatureMode.SENTIMENT, FeatureMode.TFIDF_SENTIMENT,
            FeatureMode.ST_SENTIMENT, FeatureMode.ALL,
        )
        if needs_tfidf and tfidf_embeddings is None:
            raise ValueError(f"mode='{self.mode}' requires tfidf_embeddings but got None.")
        if needs_st and st_embeddings is None:
            raise ValueError(f"mode='{self.mode}' requires st_embeddings but got None.")
        if needs_sent and feature_eng is None:
            raise ValueError(f"mode='{self.mode}' requires feature_eng but got None.")
        if needs_sent:
            for attr in self._SENTIMENT_ATTRS:
                if getattr(feature_eng, attr, None) is None:
                    raise ValueError(
                        f"feature_eng.{attr} is None. Call feature_eng.build_features() first."
                    )

    @staticmethod
    def _merge_embeddings(
        events: pd.DataFrame,
        embeddings: pd.DataFrame,
        prefix: str,
    ) -> pd.DataFrame:
        """Left-join event list with an embedding DataFrame on (filing_date, asset)."""
        emb_feat_cols = [c for c in embeddings.columns if c not in _SENTINEL_COLS]
        merged = events.merge(
            embeddings[["filing_date", "asset"] + emb_feat_cols],
            on=["filing_date", "asset"],
            how="left",
        )
        n_miss = merged[emb_feat_cols[0]].isna().sum() if emb_feat_cols else 0
        if n_miss:
            logger.warning("%s embeddings: %d/%d events had no match.", prefix, n_miss, len(events))
        return merged

    def _lookup_sentiment(
        self,
        events: pd.DataFrame,
        feature_eng: FeatureEngineering,
    ) -> pd.DataFrame:
        """
        Look up each event's sentiment values from the panel DataFrames using
        positional indexing — efficient for large event lists.
        """
        result = events[["filing_date", "asset"]].copy()

        for attr in self._SENTIMENT_ATTRS:
            panel: pd.DataFrame = getattr(feature_eng, attr)
            date_pos = panel.index.get_indexer(events["filing_date"])
            asset_pos = panel.columns.get_indexer(events["asset"])
            valid = (date_pos >= 0) & (asset_pos >= 0)

            values = np.full(len(events), np.nan)
            values[valid] = panel.values[date_pos[valid], asset_pos[valid]]
            result[attr] = values

            n_miss = (~valid).sum()
            if n_miss:
                logger.warning(
                    "Sentiment '%s': %d/%d events had no match in panel.",
                    attr, n_miss, len(events),
                )

        return result

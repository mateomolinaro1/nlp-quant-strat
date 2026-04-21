import logging
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Imports
from nlp_quant_strat.data.data_manager import DataManager
from nlp_quant_strat.data.feature_engineering import FeatureEngineering
from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.backtester.strategies import CrossSectionalPercentiles
from nlp_quant_strat.backtester.portfolio import EqualWeightingScheme
from nlp_quant_strat.backtester.backtest import Backtest
from nlp_quant_strat.backtester.analysis import PerformanceAnalyser
from nlp_quant_strat.backtester.visualization import Visualizer
from nlp_quant_strat.utils.utils import S3Utils

def cross_zscore(df):
    """Robust version of cross-sectional Z-score"""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, 1) # Avoid la division par 0
    return df.sub(mean, axis=0).div(std, axis=0)

def main():
    # 1. Configuration et Logging
    load_dotenv()
    config = Config()
    
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("FinalAnalysis")

    # 2. Loading des données
    logger.info("Loading des données from S3...")
    data_manager = DataManager(config=config)
    data_manager.load_data()

    # 3. Feature Engineering
    logger.info("Treating des features NLP...")
    feature_engineering = FeatureEngineering(data=data_manager, config=config)
    feature_engineering.build_features()

    # --- CORRECTION 1 : saving de all features implemented ---
    if config.load_or_compute_features == "compute":
        logger.info("saving all des features sur S3...")
        # On utilise la defined list dans la class FeatureEngineering
        for feat_name in feature_engineering.feature_names:
            df_to_save = getattr(feature_engineering, feat_name)
            if df_to_save is not None:
                S3Utils.upload_df_with_index(
                    df=df_to_save, 
                    bucket=config.bucket_name, 
                    path=f"data/features/{feat_name}.parquet"
                )

    # 4. Strategy (multi-factor combination)
    logger.info("Combination of signals: Polarity + Polarity_Delta")
    
    # retrieve signals
    z1 = cross_zscore(feature_engineering.polarity)
    z2 = cross_zscore(feature_engineering.polarity_delta)

    # SOMME ROBUST : On add les Z-scores
    # On replace les 0 par NaN AVANT add pour ne pas pollute les averages
    composite_signal = z1.add(z2, fill_value=0) / 2
    
    # --- LA LIGN MAGIQUE ---
    # On replace les 0 par NaN pour que les percentiles not to be computed
    # QUE sur les actions qui ont un true signal NLP ce jour-là.
    composite_signal = composite_signal.replace(0, np.nan)
    # ------------------------

    logger.info(f"Number of active signals (non-NaN) : {composite_signal.notna().sum().sum()}")

    strategy = CrossSectionalPercentiles(
        returns=data_manager.get_asset_returns(),
        signal_values=composite_signal, 
        percentiles_winsorization=config.percentiles_winsorization,
    )
    
    # 4. Strategy
    strategy.compute_signals_values()
    signals = strategy.compute_signals(
        percentiles_portfolios=config.percentiles_portfolios,
        industry_segmentation=None if config.industry_segmentation == "" else "with_industry_segmentation",
    )

    # --- ACTION : PROPAGATION DES SIGNALS ---
    # On propagate les 1 et les -1 pendant 66 jours (rebal_periods)
    # to be visible for the next rebalancing date.
    signals = signals.replace(0, np.nan).ffill(limit=config.rebal_periods).fillna(0)
    # ----------------------------------------

    # Diagnostic mis à jour
    num_buys = (signals == 1).sum().sum()
    logger.info(f"Number of buy signals (après propagation) : {num_buys}")

    if num_buys == 0:
        logger.error("❌ Strategy generated no buy. Check percentiles.")
        return
    # ------------------
    
    # 5. Construction du Ptf
    logger.info("Rebal Ptf...")
    ptf = EqualWeightingScheme(
        returns=data_manager.get_asset_returns(),
        signals=signals,
        rebal_periods=config.rebal_periods,
        portfolio_type=config.portfolio_type
    )
    
    # --- SECURITY ANTI-INDEX ERROR ---
    # On check si on a des dates de rebal valid avant de lancer
    if signals.sum(axis=1).abs().sum() == 0:
        logger.error("❌ No signal detected for the strategy. Ptf is void.")
        return
        
    ptf.compute_weights()
    ptf.rebalance_portfolio()

    # 6. Backtest
    logger.info("Launching Backtest...")
    backtester = Backtest(
        returns=data_manager.get_asset_returns(),
        weights=ptf.rebalanced_weights,
        turnover=ptf.turnover,
        transaction_costs=config.transaction_costs,
        strategy_name=f"{config.strategy_name}_COMPOSITE"
    )
    backtester.run_backtest()

    # 7. ANALYSE
    logger.info("Analyzing metrics...")
    perf_analyzer = PerformanceAnalyser(
        portfolio_returns=backtester.cropped_portfolio_net_returns,
        freq=config.market_data_frequency,
        bench_returns=data_manager.get_benchmark_returns(),
        percentiles=f"({config.percentiles_portfolios[0]}-{config.percentiles_portfolios[1]})",
        rebal_freq=f"{config.rebal_periods} days"
    )
    metrics = perf_analyzer.compute_metrics()

    # 8. Display
    print("\n" + "="*60)
    print(f"RESULTS : {config.strategy_name} (Signal: COMPOSITE)")
    print("="*60)
    # ... (Le rest de ton code de display est same)
    print(pd.DataFrame(metrics, index=[0]).T) # Display quick pour test
    print("="*60)

    # 9. Visualisation
    vizu = Visualizer(performance=perf_analyzer)
    vizu.plot_cumulative_performance(
        saving_path=config.ROOT_DIR / "outputs" / "figures" / f"{config.strategy_name}_composite_returns.png"
    )

if __name__ == "__main__":
    main()
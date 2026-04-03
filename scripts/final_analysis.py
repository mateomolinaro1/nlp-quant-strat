import logging
import sys
import pandas as pd
from dotenv import load_dotenv

# Imports de ton projet
from nlp_quant_strat.data.data_manager import DataManager
from nlp_quant_strat.data.feature_engineering import FeatureEngineering
from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.backtester.strategies import CrossSectionalPercentiles
from nlp_quant_strat.backtester.portfolio import EqualWeightingScheme
from nlp_quant_strat.backtester.backtest import Backtest
from nlp_quant_strat.backtester.analysis import PerformanceAnalyser
from nlp_quant_strat.backtester.visualization import Visualizer
from nlp_quant_strat.utils.utils import S3Utils

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

    # 2. Chargement des données (Depuis S3)
    logger.info("Chargement des données depuis S3...")
    data_manager = DataManager(config=config)
    data_manager.load_data()

    # 3. Feature Engineering (Calcul des scores NLP)
    logger.info("Traitement des features NLP...")
    feature_engineering = FeatureEngineering(data=data_manager, config=config)
    feature_engineering.build_features()

    # --- OPTIONNEL : Sauvegarde sur S3 si calculé localement pour éviter de refaire le calcul ---
    if config.load_or_compute_features == "compute":
        logger.info("Sauvegarde des nouvelles features sur S3 pour usage futur...")
        for feat_name in ["polarity", "positive_count", "negative_count", "sentiment_density", "word_count"]:
            df_to_save = getattr(feature_engineering, feat_name)
            if df_to_save is not None:
                S3Utils.upload_df_with_index(
                    df=df_to_save, 
                    bucket=config.bucket_name, 
                    path=f"data/features/{feat_name}.parquet"
                )

    # 4. Stratégie (Sélection dynamique du signal)
    # On récupère le nom du signal depuis le config.json (ex: "polarity")
    signal_name = getattr(config, "signal_feature", "polarity") 
    logger.info(f"Exécution de la stratégie utilisant le signal : {signal_name}")
    
    signal_data = getattr(feature_engineering, signal_name)
    
    strategy = CrossSectionalPercentiles(
        returns=data_manager.get_asset_returns(),
        signal_values=signal_data,
        percentiles_winsorization=config.percentiles_winsorization,
    )
    
    # Calcul des valeurs (Z-scores + Winsorization)
    strategy.compute_signals_values()
    
    # Génération des signaux finaux (Long/Short/Neutre)
    signals = strategy.compute_signals(
        percentiles_portfolios=config.percentiles_portfolios,
        industry_segmentation=None if config.industry_segmentation == "" else "with_industry_segmentation",
    )

    # 5. Construction du Portefeuille (Equipondéré)
    logger.info("Rebalancement du portefeuille...")
    ptf = EqualWeightingScheme(
        returns=data_manager.get_asset_returns(),
        signals=signals,
        rebal_periods=config.rebal_periods,
        portfolio_type=config.portfolio_type
    )
    ptf.compute_weights()
    ptf.rebalance_portfolio()

    # 6. Backtest (Calcul des rendements nets)
    logger.info("Lancement du Backtest...")
    backtester = Backtest(
        returns=data_manager.get_asset_returns(),
        weights=ptf.rebalanced_weights,
        turnover=ptf.turnover,
        transaction_costs=config.transaction_costs,
        strategy_name=config.strategy_name
    )
    backtester.run_backtest()

    # 7. ANALYSE DES PERFORMANCES
    logger.info("Analyse des métriques de performance...")
    perf_analyzer = PerformanceAnalyser(
        portfolio_returns=backtester.cropped_portfolio_net_returns,
        freq=config.market_data_frequency,
        bench_returns=data_manager.get_benchmark_returns(),
        percentiles=f"({config.percentiles_portfolios[0]}-{config.percentiles_portfolios[1]})",
        rebal_freq=f"{config.rebal_periods} days"
    )

    metrics = perf_analyzer.compute_metrics()

    # 8. Affichage des résultats
    comparison_df = pd.DataFrame({
        'Métrique': ['Total Return', 'Ann. Return', 'Ann. Volatility', 'Sharpe Ratio', 'Max Drawdown'],
        'Stratégie (NLP)': [
            metrics['total_return'], 
            metrics['annualized_return'], 
            metrics['annualized_volatility'], 
            metrics['annualized_sharpe_ratio'], 
            metrics['max_drawdown']
        ],
        'Benchmark': [
            metrics['total_return_bench'], 
            metrics['annualized_return_bench'], 
            metrics['annualized_volatility_bench'], 
            metrics['annualized_sharpe_ratio_bench'], 
            metrics['max_drawdown_bench']
        ]
    })

    print("\n" + "="*60)
    print(f"RÉSULTATS : {config.strategy_name} (Signal: {signal_name})")
    print("="*60)
    print(comparison_df.to_string(index=False))
    print("="*60)

    # 9. Visualisation
    logger.info("Génération du graphique final...")
    vizu = Visualizer(performance=perf_analyzer)
    vizu.plot_cumulative_performance(
        saving_path=config.ROOT_DIR / "outputs" / "figures" / f"{config.strategy_name}_{signal_name}_returns.png"
    )
    logger.info(f"Analyse terminée. Fichier sauvegardé.")

if __name__ == "__main__":
    main()
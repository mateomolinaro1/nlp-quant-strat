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

    # 2. Chargement des données (Depuis S3 comme configuré)
    logger.info("Chargement des données depuis S3...")
    data_manager = DataManager(config=config)
    data_manager.load_data()

    # 3. Feature Engineering (Calcul des scores NLP)
    logger.info("Calcul des features (NLP signals)...")
    feature_engineering = FeatureEngineering(data=data_manager, config=config)
    feature_engineering.build_features()

    # 4. Stratégie (Tri par Percentiles)
    logger.info("Exécution de la stratégie Cross-Sectional...")
    strategy = CrossSectionalPercentiles(
        returns=data_manager.get_asset_returns(),
        signal_values=feature_engineering.positive_count, # Ton signal de sentiment
        percentiles_winsorization=config.percentiles_winsorization,
    )
    strategy.compute_signals_values()
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

    # 7. ANALYSE DES PERFORMANCES (La partie corrigée)
    logger.info("Analyse des métriques de performance...")
    # On crée l'INSTANCE 'perf_analyzer'
    perf_analyzer = PerformanceAnalyser(
        portfolio_returns=backtester.cropped_portfolio_net_returns,
        freq=config.market_data_frequency,
        bench_returns=data_manager.get_benchmark_returns(),
        percentiles=f"({config.percentiles_portfolios[0]}-{config.percentiles_portfolios[1]})",
        rebal_freq=f"{config.rebal_periods} days"
    )

    # Appel de la méthode sur l'INSTANCE
    metrics = perf_analyzer.compute_metrics()

    # 8. Affichage des résultats sous forme de tableau
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

    print("\n" + "="*50)
    print("RECAPITULATIF DES PERFORMANCES")
    print("="*50)
    print(comparison_df.to_string(index=False))
    print("="*50)

    # 9. Visualisation (Optionnel)
    logger.info("Génération du graphique final...")
    vizu = Visualizer(performance=perf_analyzer)
    vizu.plot_cumulative_performance(
        saving_path=config.ROOT_DIR / "outputs" / "figures" / f"{config.strategy_name}_final_plot.png"
    )
    logger.info(f"Analyse terminée. Graphique sauvegardé dans outputs/figures/")

if __name__ == "__main__":
    main()
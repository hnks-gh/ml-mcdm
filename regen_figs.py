#!/usr/bin/env python
"""Quick script to regenerate ranking figures with all 6 methods."""

import sys
import logging
from pipeline import DataPipeline
from ranking.hierarchical_pipeline import HierarchicalRankingPipeline
from output.visualization import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading panel data...")
    dp = DataPipeline()
    panel = dp.load_panel()
    
    logger.info("Setting up ranking pipeline...")
    ranking_pipe = HierarchicalRankingPipeline()
    
    # Get CRITIC weights
    from weighting.critic_weighting import HierarchicalCRITICWeighter
    weighter = HierarchicalCRITICWeighter()
    weight_result = weighter.fit(panel)
    
    logger.info("Running ranking for latest year...")
    latest_year = max(panel.years)
    
    # Get weights
    sc_weights = weight_result.subcriteria_weights_all_years.get(latest_year, {})
    crit_weights = weight_result.criterion_weights_all_years.get(latest_year, {})
    
    result = ranking_pipe.rank(
        panel,
        subcriteria_weights=sc_weights,
        target_year=latest_year,
        criterion_weights=crit_weights,
    )
    
    logger.info(f"Ranking completed for {latest_year}")
    logger.info(f"Methods in result: {list(result.criterion_method_scores.get(list(result.criterion_method_scores.keys())[0], {}).keys())}")
    
    logger.info("Generating visualizations...")
    viz = Visualizer(panel)
    
    # Generate fig08e and fig08f
    path_e = viz.mcdm.plot_method_stability_comparison(result)
    logger.info(f"fig08e saved to: {path_e}")
    
    path_f = viz.mcdm.plot_method_disc_power_comparison(result)
    logger.info(f"fig08f saved to: {path_f}")
    
    logger.info("Done!")

if __name__ == '__main__':
    main()

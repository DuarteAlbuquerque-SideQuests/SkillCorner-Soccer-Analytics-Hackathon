# SkillCorner-Soccer-Analytics-Hackathon
A source-to-sink maximum flow framework for football analytics. Using the Ford-Fulkerson algorithm and tracking data, this project quantifies continuous attacking capacity and potential threat at a millisecond resolution. Winner: 2nd Place at the 2026 Soccer Analytics Hackathon, ETH Zurich (Partners: FIFA, DFB, and SkillCorner).


# Football Is Flow: Dynamic Max-Flow Network Analysis

**2nd Place Winner – Soccer Analytics Hackathon (ETH Zurich, 2026)**
*Developed by: Duarte Albuquerque, Elijah Tamarchenko, Juan Carlos Meyer, and Bruno Croso*

## Overview
Traditional football analytics frequently rely on discrete, event-based models that evaluate player actions only at the moment of execution. This project introduces a continuous modeling approach by treating the attacking team as a living, directed capacity network. By applying network flow algorithms to high-resolution tracking data, we quantify the volume of goal-scoring threat created at every millisecond of a match.

## Technical Framework
The model utilizes a source-to-sink architecture to evaluate tactical value:
* **Network Topology:** The ball carrier acts as the source, with the opponent's goal defined as the sink.
* **Edge Calibration:** Pass and run edges are weighted using a utility model that balances expected gain against the cost of turnover, adjusted for defensive pressure and spatial geometry.
* **Flow Capacity:** Using the Ford-Fulkerson algorithm, the model produces a maximum flow scalar per frame, representing the total pressure exerted by the attacking side.
* **Regret Analysis:** We quantify suboptimality by measuring the "regret" between the maximum available flow and the utility of the action actually taken.

## Repository Contents
* `interactive_dashboard.py`: A Dash and Plotly-based visualization engine for real-time analysis of threat channels and flow intensity.
* `network flow visualization v3.py`: The core computational pipeline for network modeling and precomputation.
* `DASHBOARD_GUIDE.md`: Technical instructions for environment setup and data ingestion.
* `Submission.pdf`: Comprehensive documentation regarding the mathematical methodology and period-aware normalization.

## Data Privacy and Usage
To comply with Licensing Agreements, raw proprietary tracking and event data have been removed from this repository. Users must provide their own tracking data in the directory structure specified in the documentation to utilize the analysis scripts.

## Conclusion
This framework represents a shift from reactive metrics toward the mapping of continuous spatial geometry and potential threat. We believe the future of the discipline lies in quantifying not only the actions taken, but the full landscape of possibilities available at any moment in the game.

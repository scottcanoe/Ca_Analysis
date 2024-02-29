## Ca_Analysis: Calcium imaging suite for dataset management, preprocessing pipelines, and analysis.

This is the codebase used in the Gavornik Lab for working with imaging data in the [Gavornik Lab](https://gavorniklab.bu.edu/). It handles file management, preprocessing, and common analysis tools. Pipelines and experiment-specific scripts are located in the experiments directory (i.e., experiments/seq_learn_3 contains gobs of processing and analysis scripts). The core package (`ca_analysis`) contains commonly used objects and processing utilities, with `session.py` containing probably the most important interfaces. The architecture is extensible and has been used in a variety of recording modalities, including electrophysiology.

I'm currently in the process of reorganizing and documenting this project to make it useful for others. Stay tuned.

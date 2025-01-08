# ML Benchmarking

We evaluate TableMage ConversationalAgent on a set of tabular datasets for machine learning.
We obtain datasets from the benchmark curated by Grinsztajin et al. for their paper
"Why do tree-based models still outperform deep learning on tabular data?" published in 2022.
The datasets are publically available on OpenML. 
We only consider datasets with no more than 10000 rows and 100 columns.

## Files

1. `./benchmarking/ml/get_datasets.py`: script for downloading datasets from OpenML to local computer.
2. `./benchmarking/ml/tablemage_results.py`: script for producing TableMage ConversationalAgent results.
3. `./benchmarking/ml/sklearn_results.py`: script for producing baseline results (linear models, random forest).



# DataAnalysisQA

We curated a benchmark for agentic data analysis.
This work is inspired by the InfiAgent-DABench, the first-of-its-kind benchmark for 
evaluating data analysis agents (Hu et al., 2024).
This benchmark has some key differences compared to InfiAgent-DABench.

1. **Human-curated datasets/questions and human-generated labels.**
We wrote questions by hand and generated results manually. The InfiAgent-DABench used ChatGPT (GPT-3.5 and GPT-4) to write questions and generate human labels. 

2. **Reduced technical language.** 
Questions appear less technical. For example, DataAnalysisQA would ask "Is there a statistically significant difference in mean __ between the different categories of __?" rather than "Perform a one-way ANOVA to determine if __". All statistical testing questions elicit yes-or-no responses to allow for flexibility in method/test choice. 

3. **Emphasis on persisted data transformation.** 
DataAnalysisQA contains several questions asking for data transformation. These transformations must be maintained for subsequent questions.


## Files

1. `./benchmarking/dataanalysisqa/dataanalysisqa.tsv`: 
Benchmarking questions with categories.

2. `./benchmarking/dataanalysisqa/generate_labels.py`:
Script to produce labels/ground truth. Also see `./benchmarking/dataanalysisqa/label_scripts`.

3. `./benchmarking/dataanalysisqa/tablemage_results.py`: Script to reproduce TableMage ConversationalAgent results. NOTE: We used cloud compute providers for LLM inference. As such, you may obtain different results, as we have no control over model deprecations/updates and random seeds.

4. `./benchmarking/dataanalysisqa/reformat_result.py`: 
Script for reformatting answers. If you want to benchmark your own agent, you simply need to add a directory in the results folder, following the structure detailed in `./benchmarking/dataanalysisqa/tablemage_results.py`. 
NOTE: This script uses an LLM to parse answers and structure them correctly. LLMs can make mistakes

5. `./benchmarking/dataanalysisqa/datasets`:
Datasets in the benchmark.


## Rules

1. Use an 80-20 train-test split, split using scikit-learn's train_test_split function with random seed 42. The agent must be instructed to do so if this behavior is not inherent.

2. If explicitly asked to transform the dataset (e.g., scaling, imputation, feature engineering), keep the changes for future questions. That is, changes made to the dataset must be persisted for future responses.

3. When transforming data (e.g., feature scaling), always fit on the train dataset and transform the test dataset based on the train dataset.

4. When fitting models, always fit on the train dataset and predict on the test dataset.

5. For exploratory analysis (e.g., statistical testing, summary statistics), always consider the entire dataset.

6. Temporarily drop rows with missing values in variables of interest prior to each analysis step.

7. Round numeric answers to 3 decimal places.

8. Use significance level 0.05 for statistical tests.


### Example preamble

We provided the following string to GPT-4o Advanced Data Analysis immediately upon dataset upload:
```
I am attaching a dataset. I will then give you instructions for analyzing the dataset. Here are some rules: (1) Immediately split the dataset into 80/20 train/test sets using sklearn’s train_test_split function, with random seed 42. (2) If you are explicitly asked to transform the dataset (e.g., scaling, imputation, feature engineering), keep the changes for future questions. (3) When transforming data (e.g., feature scaling), always fit on the train dataset and transform the test dataset based on the train dataset. (4) When fitting models, always fit on the train dataset and predict on the test dataset. (5) For exploratory analysis (e.g., statistical testing, summary statistics), always consider the entire dataset. (6) Temporarily drop rows with missing values in variables of interest prior to each analysis step. (7) Return a sentence for each query describing your findings, round numeric answers to 3 decimal places. (8) Use significance level 0.05 for statistical tests.
```

This can be modified into a system prompt as needed. Not all agents need this degree of specification in their system prompts, such as the case where an agent's tools automatically perform the train-test split.


### Other constraints

1. For correlation analysis, use the Pearson correlation (we found current LLMs tend to do this automatically).


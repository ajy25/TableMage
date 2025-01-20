from pathlib import Path
import pandas as pd
import time
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate


curr_dir = Path(__file__).resolve().parent


results_folder = curr_dir / "results" / "tablemage_llama3.3_70b"  # change appropriately

questions_df = pd.read_csv(curr_dir / "dataanalysisqa.tsv", sep="\t")
results_unformatted_df = pd.read_csv(results_folder / "answers.csv")


joined_df = questions_df.merge(results_unformatted_df, on="Question ID")


# read in the API key
api_key = curr_dir / "api_key.txt"
with open(api_key, "r") as f:
    api_key = f.read().strip()

llm = OpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key)


initial_prompt_template = PromptTemplate(
    template="""\
Someone answered a question about a dataset. \
Your job is to format their answer in a structured manner. \
Below, I will provide you with the question, the unformatted answer, and a list of keywords. \
Your job is to use the keywords to format the answer in a structured manner. \
Round all numerical answers to 3 decimal places. \
If an answer is missing, replace it with "MISSING". \
See the examples below.

-- Example 1 --
Question: What is the average age of the passengers?
Unformatted Answer: The average age of the passengers is 29.7.
Keywords: mean

Model Response: mean=29.700

-- Example 2 --
Question: Find the median and standard deviation of the house prices.
Unformatted Answer: The median house price is $300,000 and the standard deviation is $50,000.
Keywords: median, standard_deviation

Model Response: median=300000.000, standard_deviation=50000.000

-- Example 3 --
Question: Is there a significant correlation between age and income? What is the correlation coefficient and p-value?
Unformatted Answer: There is a significant correlation between age and income (p-value = 2.214e-7). The correlation coefficient is 0.5216.
Keywords: yes_or_no, correlation_coefficient, p_value

Model Response: yes_or_no=yes, correlation_coefficient=0.522, p_value=0.000

-- Example 4 --
Question: What is the average age of the passengers?
Unformatted Answer: Error occured: Age column is missing.
Keywords: mean

Model Response: mean=MISSING

-- Your Turn --
Question: {question}
Unformatted Answer: {answer}
Keywords: {keywords}

Model Response: 
"""
)


validation_prompt_template = PromptTemplate(
    template="""\
I am going to present you with a list of keywords with their corresponding values. \
Then, I will provide you with a correct list of keywords. \
Your job is to filter the first list of keywords based on the second list of keywords. \
Also, please ensure that all numbers are rounded to 3 decimal places-- 
regardless of whether they are integers or floats. \
Let me show you an example.

-- Example 1 --
Input: mean=29.700, median=300000.000, standard_deviation=50000.000
Keywords: mean, standard_deviation

Model Response: mean=29.700, standard_deviation=50000.000

-- Example 2 --
Input: n_examples=98
Keywords: n_examples

Model Response: n_examples=98.000

-- Your Turn --
Input: {input}
Keywords: {keywords}

Model Response:
"""
)


question_ids = []
formatted_answers = []


for row in joined_df.iterrows():
    question_id = row[1]["Question ID"]
    question = row[1]["Question"]
    dataset = row[1]["Dataset"]
    answer = row[1]["Unformatted Answer"]
    output_keywords_list = row[1]["Output Keywords"].split("; ")
    print(f"Question ID: {question_id}")
    print(f"Question: {question}")
    print(f"Dataset: {dataset}")
    print(f"Output Keywords: {output_keywords_list}")
    print(f"Answer: {answer}")

    output = llm.complete(
        initial_prompt_template.format(
            question=question, answer=answer, keywords=", ".join(output_keywords_list)
        )
    )
    output = str(output)
    print(f"Formatted Output: {output}")

    output = llm.complete(
        validation_prompt_template.format(
            input=output, keywords=", ".join(output_keywords_list)
        )
    )
    output = str(output)
    print(f"Validation Output: {output}")
    print("\n\n")

    question_ids.append(question_id)
    formatted_answers.append(output)

    time.sleep(2)


formatted_answers_df = pd.DataFrame(
    {"Question ID": question_ids, "Formatted Answer": formatted_answers}
)
formatted_answers_df.to_csv(
    results_folder / "formatted_answers.tsv", sep="\t", index=False
)

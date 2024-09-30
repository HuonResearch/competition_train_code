import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from llms.llms import MODELS
from setup import *

def format_prompt(prompt: str, benchmark_contents: str) -> str:
    """
    Format a user submitted prompt to contain the appropriate contents from the eval datasets.

    :param prompt: User-submitted prompt, where the location for the eval datasets contents in indicated by the string "{CONTENT}".
    :param benchmark_contents: Content drawn from the evaluation datasets
    :return: Formatted, complete string
    """
    if "{CONTENT}" not in prompt:
        print("Incomplete prompt; missing replacement indicator")
    return prompt.replace("{CONTENT}", benchmark_contents)

def main(prompt: str, model:str, benchmark_eval: pd.DataFrame) -> float:
    """
    Run all prompt-model submissions against the benchmark evaluation dataset.

    :param prompt: A string containing the prompt to submit to the LLM. The prompt must include the string "{CONTENT}" where the contents from the training dataset is meant to be inserted.
    :param model: A string indicating the selection of the
    :param benchmark_eval: Benchmarking evaluation dataset containing `content` and `label` columns.
    :return: Accuracy score corresponding to the model-prompt combination's performance.
    """

    # Create copy of eval datasets to store results
    this_benchmark_output = benchmark_eval.copy()

    # Iterate through each row of training dataset
    for j in tqdm(benchmark_eval.index):
        # Format submitted prompt with benchmark datasets
        user_prompt = format_prompt(prompt, str(benchmark_eval.loc[j, CONTENT_COL]))

        # Assign user selected model to correct LLM processing class
        if model not in MODELS.keys():
            raise KeyError("Invalid LLM. Please check that your LLM name is formatted correctly.")
        user_model = MODELS[model]

        # Extract LLM responses; list of len 3
        llm_responses = user_model.submit(user_prompt)

        # Loop through LLM responses and see if any match desired output. If yes, then record the one that matches.If  none do, output is assigned to first response
        llm_label = llm_responses[0]
        for response in llm_responses:
            if response == str(benchmark_eval.loc[j, LABEL_COL]):
                llm_label = response

        # Add in assessment info to original df
        this_benchmark_output.loc[j, "llm_label"] = llm_label

    # Assess this submission; compared `desired_outcome` to `outcome`
    accuracy = accuracy_score(this_benchmark_output[LABEL_COL].astype(str), this_benchmark_output['llm_label'].astype(str))

    # Output accuracy score
    return accuracy


if __name__ == "__main__":

    # Import train dataset
    benchmark_train = pd.read_csv(os.path.join("datasets", TRAIN_FILE)).head(100)

    # Write your prompt and model HERE
    # `prompt`: must contain the string "{CONTENT}" where you wish the content from the training dataset to be inserted
    # `model`: must match the formatting of one of the MODELS dictionary keys from llms/llms.py
    model = "Cohere Command R"
    prompt = "How many R's are in the following word? \"{CONTENT}\""

    # Evaluate model-prompt against the
    submission_accuracy = main(prompt, model, benchmark_train)

    # Print results
    print(f"Prompt: {prompt}\nModel: {model}\nAccuracy score (training data): {round(submission_accuracy, 3)}")

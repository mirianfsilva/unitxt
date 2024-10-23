import os
import tempfile

import pandas as pd

from unitxt import add_to_catalog, evaluate, get_logger, load_dataset, register_local_catalog
from unitxt.inference import HFPipelineBasedInferenceEngine
from unitxt.templates import MultipleChoiceTemplate, TemplatesList
from unitxt.text_utils import print_dict

logger = get_logger()
# PROMPT-EVAL PROMPTS VARIATIONS
templates = {
    "prompteval_with_topic": {
        "mmlu": {
            "1": "Below are several multiple-choice questions concerning {topic}.\n{question}.\nOptions:\n{choices}.\nCorrect answer:",
            "2": "Presented here are multiple-choice queries related to {topic}.\n{question}.\nAnswer choices:\n{choices}.\nSolution:",
            "3": "Listed below are questions in a multiple-choice format about {topic}.\n{question}.\nChoices:\n{choices}.\nCorrect answer:",
            "4": "Here are several questions with multiple-choice answers regarding {topic}.\n{question}.\nPossible answers:\n{choices}.\nFinal answer:",
            "5": "The following contains multiple-choice questions on the topic of {topic}.\n{question}.\nSelections:\n{choices}.\nCorrect response:",
            "6": "These are multiple-choice items on the subject of {topic}.\n{question}.\nAnswer options:\n{choices}.\nChosen answer:",
            "7": "Here are some multiple-choice questions with answers about {topic}.\n{question}.\nPossible responses:\n{choices}.\nFinal response:",
            "8": "The questions below are multiple-choice, focusing on {topic}.\n{question}.\nAnswer selections:\n{choices}.\nChosen response:",
            "9": "Below is a series of multiple-choice questions covering {topic}.\n{question}.\nAnswer options:\n{choices}.\nSolution:",
            "10": "The following is a set of multiple-choice questions related to {topic}.\n{question}.\nChoices available:\n{choices}.\nCorrect selection:",
            "11": "Below are MCQs on the subject of {topic}.\n{question}.\nAvailable answers:\n{choices}.\nSelected answer:",
            "12": "Here is a list of multiple-choice questions that cover {topic}.\n{question}.\nResponse options:\n{choices}.\nAnswer:",
            "13": "The following consists of multiple-choice questions concerning {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nRight answer:",
            "14": "Presented below are multiple-choice questions related to {topic}.\n{question}.\nAnswer selections:\n{choices}.\nChosen answer:",
            "15": "Listed here are multiple-choice questions on the topic of {topic}.\n{question}.\nOptions for answers:\n{choices}.\nCorrect response:",
            "16": "These are MCQs on {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nSolution:",
            "17": "Presented here are multiple-choice questions for {topic}.\n{question}.\nResponse options:\n{choices}.\nFinal answer:",
            "18": "Below is a list of multiple-choice questions related to {topic}.\n{question}.\nPossible answers:\n{choices}.\nAnswer:",
            "19": "Here is a series of multiple-choice queries concerning {topic}.\n{question}.\nAnswer options:\n{choices}.\nCorrect answer:",
            "20": "Below are multiple-choice questions regarding {topic}.\n{question}.\nAnswer choices:\n{choices}.\nSelected answer:",
            "21": "These are some multiple-choice questions about {topic}.\n{question}.\nAvailable answers:\n{choices}.\nSolution:",
            "22": "Presented are multiple-choice questions on {topic}.\n{question}.\nAnswer options:\n{choices}.\nChosen answer:",
            "23": "These multiple-choice questions pertain to {topic}.\n{question}.\nAvailable responses:\n{choices}.\nFinal answer:",
            "24": "The following are multiple-choice queries based on {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nSelected answer:",
            "25": "The questions below are multiple-choice and focused on {topic}.\n{question}.\nAnswer choices:\n{choices}.\nCorrect response:",
            "26": "Below is a collection of multiple-choice questions about {topic}.\n{question}.\nAnswer options:\n{choices}.\nChosen answer:",
            "27": "These MCQs are related to {topic}.\n{question}.\nAnswer selections:\n{choices}.\nCorrect answer:",
            "28": "The following is a set of multiple-choice questions for {topic}.\n{question}.\nChoices available:\n{choices}.\nFinal response:",
            "29": "Here are multiple-choice questions based on {topic}.\n{question}.\nPossible answers:\n{choices}.\nSolution:",
            "30": "The below multiple-choice questions focus on {topic}.\n{question}.\nAvailable answers:\n{choices}.\nAnswer:",
            "31": "The following contains MCQs concerning {topic}.\n{question}.\nAnswer selections:\n{choices}.\nChosen response:",
            "32": "These are multiple-choice questions for {topic}.\n{question}.\nAnswer choices:\n{choices}.\nCorrect response:",
            "33": "Listed below are multiple-choice questions related to {topic}.\n{question}.\nPossible responses:\n{choices}.\nAnswer:",
            "34": "Below are multiple-choice questions concerning {topic}.\n{question}.\nChoices:\n{choices}.\nFinal answer:",
            "35": "Presented here are MCQs about {topic}.\n{question}.\nResponse options:\n{choices}.\nSolution:",
            "36": "These are multiple-choice questions regarding {topic}.\n{question}.\nAnswer selections:\n{choices}.\nChosen answer:",
            "37": "The following is a list of MCQs related to {topic}.\n{question}.\nChoices available:\n{choices}.\nCorrect response:",
            "38": "Here is a set of multiple-choice questions on {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nSelected answer:",
            "39": "These questions are in multiple-choice format, concerning {topic}.\n{question}.\nAnswer choices:\n{choices}.\nAnswer:",
            "40": "The following multiple-choice questions focus on {topic}.\n{question}.\nAnswer options:\n{choices}.\nSolution:",
            "41": "Presented below are multiple-choice queries about {topic}.\n{question}.\nAvailable answers:\n{choices}.\nCorrect answer:",
            "42": "Below is a list of MCQs related to {topic}.\n{question}.\nPossible answers:\n{choices}.\nFinal response:",
            "43": "These are questions with multiple-choice answers related to {topic}.\n{question}.\nAnswer selections:\n{choices}.\nChosen answer:",
            "44": "Listed here are multiple-choice questions about {topic}.\n{question}.\nAvailable options:\n{choices}.\nCorrect response:",
            "45": "The questions below are multiple-choice and pertain to {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nSelected answer:",
            "46": "The following is a collection of multiple-choice questions related to {topic}.\n{question}.\nAnswer options:\n{choices}.\nAnswer:",
            "47": "These multiple-choice questions are about {topic}.\n{question}.\nChoices available:\n{choices}.\nSolution:",
            "48": "Presented below are MCQs for {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nChosen response:",
            "49": "These questions in multiple-choice format are about {topic}.\n{question}.\nAnswer selections:\n{choices}.\nFinal answer:",
            "50": "The following consists of multiple-choice queries on {topic}.\n{question}.\nChoices:\n{choices}.\nCorrect answer:",
            "51": "Here is a list of multiple-choice questions that relate to {topic}.\n{question}.\nAnswer options:\n{choices}.\nSelected answer:",
            "52": "Listed below are multiple-choice questions focusing on {topic}.\n{question}.\nChoices available:\n{choices}.\nCorrect response:",
            "53": "Presented are multiple-choice questions on the topic of {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nChosen answer:",
            "54": "Below is a set of multiple-choice questions regarding {topic}.\n{question}.\nAnswer choices:\n{choices}.\nFinal response:",
            "55": "These are multiple-choice questions focusing on {topic}.\n{question}.\nAnswer selections:\n{choices}.\nSolution:",
            "56": "The following questions are in multiple-choice format and cover {topic}.\n{question}.\nAvailable responses:\n{choices}.\nCorrect answer:",
            "57": "Listed here are MCQs for the subject of {topic}.\n{question}.\nPossible answers:\n{choices}.\nAnswer:",
            "58": "Below is a list of multiple-choice questions that concern {topic}.\n{question}.\nChoices:\n{choices}.\nSelected answer:",
            "59": "Here are some multiple-choice questions that cover {topic}.\n{question}.\nAnswer options:\n{choices}.\nFinal response:",
            "60": "The following contains multiple-choice queries on {topic}.\n{question}.\nAnswer selections:\n{choices}.\nCorrect answer:",
            "61": "These multiple-choice questions are related to the subject of {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nChosen response:",
            "62": "Listed below are questions in a multiple-choice format concerning {topic}.\n{question}.\nChoices available:\n{choices}.\nFinal response:",
            "63": "Here is a series of multiple-choice queries related to {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nAnswer:",
            "64": "These MCQs focus on {topic}.\n{question}.\nAnswer choices:\n{choices}.",
            "65": "Below are multiple-choice questions about {topic}.\n{question}.\nResponse options:\n{choices}.\nCorrect answer:",
            "66": "The following is a list of multiple-choice queries concerning {topic}.\n{question}.\nAnswer options:\n{choices}.\nChosen response:",
            "67": "These are questions with multiple-choice answers about {topic}.\n{question}.\nChoices available:\n{choices}.\nFinal answer:",
            "68": "Below is a set of multiple-choice questions on {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nSolution:",
            "69": "Presented here are multiple-choice questions concerning {topic}.\n{question}.\nAnswer selections:\n{choices}.\nCorrect answer:",
            "70": "These multiple-choice questions cover {topic}.\n{question}.\nAnswer options:\n{choices}.\nFinal response:",
            "71": "The following consists of multiple-choice questions regarding {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nChosen answer:",
            "72": "Here are several multiple-choice questions on the subject of {topic}.\n{question}.\nChoices available:\n{choices}.\nCorrect response:",
            "73": "Listed here are multiple-choice questions related to {topic}.\n{question}.\nAvailable responses:\n{choices}.\nSelected answer:",
            "74": "These are MCQs about {topic}.\n{question}.\nAnswer choices:\n{choices}.\nFinal response:",
            "75": "Below is a series of multiple-choice questions about {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nCorrect answer:",
            "76": "Presented below are multiple-choice questions focusing on {topic}.\n{question}.\nChoices:\n{choices}.\nSolution:",
            "77": "The following contains multiple-choice queries concerning {topic}.\n{question}.\nAnswer options:\n{choices}.\nChosen response:",
            "78": "These MCQs are related to {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nFinal answer:",
            "79": "The questions below are in multiple-choice format and cover {topic}.\n{question}.\nAnswer selections:\n{choices}.\nCorrect answer:",
            "80": "Below is a set of multiple-choice questions about {topic}.\n{question}.\nAnswer choices:\n{choices}.\nFinal response:",
            "81": "The following is a series of MCQs related to {topic}.\n{question}.\nChoices available:\n{choices}.\nChosen answer:",
            "82": "These are multiple-choice questions focusing on {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nSolution:",
            "83": "Listed below are multiple-choice queries about {topic}.\n{question}.\nAnswer selections:\n{choices}.\nCorrect response:",
            "84": "Below is a collection of multiple-choice questions related to {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nChosen answer:",
            "85": "Presented here are multiple-choice questions for {topic}.\n{question}.\nAvailable options:\n{choices}.\nCorrect answer:",
            "86": "The following are multiple-choice questions covering {topic}.\n{question}.\nAnswer options:\n{choices}.\nFinal response:",
            "87": "These are MCQs about the subject of {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nSolution:",
            "88": "The questions below are multiple-choice and concern {topic}.\n{question}.\nAvailable responses:\n{choices}.\nChosen answer:",
            "89": "Listed here are multiple-choice queries for {topic}.\n{question}.\nAnswer choices:\n{choices}.\nCorrect response:",
            "90": "These multiple-choice questions pertain to the subject of {topic}.\n{question}.\nAnswer selections:\n{choices}.\nSelected answer:",
            "91": "Below is a list of MCQs that focus on {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nFinal answer:",
            "92": "Here are several multiple-choice questions on the topic of {topic}.\n{question}.\nAnswer options:\n{choices}.\nSolution:",
            "93": "The following contains multiple-choice queries on {topic}.\n{question}.\nChoices available:\n{choices}.\nCorrect response:",
            "94": "Presented here are multiple-choice questions concerning {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nChosen answer:",
            "95": "Listed below are MCQs focusing on {topic}.\n{question}.\nAnswer selections:\n{choices}.\nFinal response:",
            "96": "These are questions in multiple-choice format about {topic}.\n{question}.\nAnswer choices:\n{choices}.\nSolution:",
            "97": "Below is a collection of multiple-choice questions concerning {topic}.\n{question}.\nAnswer possibilities:\n{choices}.\nCorrect answer:",
            "98": "Presented here are multiple-choice questions for the topic of {topic}.\n{question}.\nChoices:\n{choices}.\nFinal response:",
            "99": "The following are multiple-choice questions that relate to {topic}.\n{question}.\nAnswer options:\n{choices}.\nChosen answer:",
            "100": "These MCQs concern the subject of {topic}.\n{question}.\nAnswer selections:\n{choices}.\nFinal answer:",
        },
    }
}


# Register a local catalog
def create_path_and_register_as_local_catalog(path):
    if not os.path.exists(path):
        os.mkdir(path)
    register_local_catalog(path)
    return path


catalog_dir = tempfile.gettempdir()  # You can replace with any fixed directory
my_catalog_dir = create_path_and_register_as_local_catalog(catalog_dir)

template_handles_prompteval = []

for index, input_format in enumerate(templates["prompteval_with_topic"]["mmlu"].values()):
    template = MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.first_character"],
    )
    template_handle = f"templates.qa.multiple_choice.prompteval_with_topic.mmlu.{index}"
    template_handles_prompteval.append(template_handle)
    add_to_catalog(template, template_handle, catalog_path=my_catalog_dir, overwrite=True)

add_to_catalog(
    artifact=TemplatesList(template_handles_prompteval),
    name="templates.qa.multiple_choice.with_topic.prompteval",
    overwrite=True,
)

card = "cards.mmlu.abstract_algebra"
model_name = "google/flan-t5-xxl"
inference_model = HFPipelineBasedInferenceEngine(model_name=model_name, max_new_tokens=32)

df = pd.DataFrame(columns=["template", "num_demos", "accuracy", "f1_micro", "ci_low", "ci_high"])

for template in template_handles_prompteval:
    dataset = load_dataset(
        card=card,
        template=template,
        num_demos=2,
        demos_pool_size=50,
        loader_limit=100,
        max_test_instances=50,
    )

    test_dataset = dataset["test"]

    predictions = inference_model.infer(test_dataset)
    evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

    logger.info(f"Sample input and output for template '{template}':")
    print_dict(
        evaluated_dataset[0],
        keys_to_print=[
            "source",
            "prediction",
            "processed_prediction",
        ],
    )

    global_scores = evaluated_dataset[0]["score"]["global"]
    print_dict(
        global_scores,
        keys_to_print=["score_name", "score", "score_ci_low", "score_ci_high"],
    )

    df.loc[len(df)] = [
        template,
        2,
        global_scores["accuracy"],
        global_scores["score"],
        global_scores["score_ci_low"],
        global_scores["score_ci_high"],
    ]

df = df.round(decimals=2)
logger.info(df.to_markdown())

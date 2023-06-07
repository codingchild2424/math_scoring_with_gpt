import openai
import os

from dotenv import dotenv_values
from tree_of_thoughts import OpenAILanguageModel
from tree_of_thoughts import MonteCarloTreeofThoughts


config = dotenv_values("./../.env")

openai.organization = config.get('OPENAI_ORGANIZATION')
openai.api_key = config.get('OPENAI_API_KEY')

##############################################################
# Normal GPT Call
##############################################################
def gpt_call(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                    {"role": "user", "content": prompt},
                ]
    )
    output_text = response["choices"][0]["message"]["content"]

    return output_text

##############################################################
# Tree of Thought GPT Call
##############################################################
def gpt_call_tree_of_thought(
        prompt,
        num_thoughts=1,
        max_steps=3,
        max_states=4,
        pruning_threshold=0.5
        ):

    model = OpenAILanguageModel(
        api_key=config.get('OPENAI_API_KEY'),
        api_model="gpt-4",
    )
    tree_of_thoughts = MonteCarloTreeofThoughts(model)

    output_text = tree_of_thoughts.solve(
        initial_prompt=prompt,
        num_thoughts=num_thoughts,
        max_steps=max_steps,
        max_states=max_states,
        pruning_threshold=pruning_threshold
    )

    print("Ouput Text: ", output_text)
    

    return output_text


import openai
import os

from dotenv import dotenv_values

config = dotenv_values("./../.env")

openai.organization = config.get('OPENAI_ORGANIZATION')
openai.api_key = config.get('OPENAI_API_KEY')


def gpt_call(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                    {"role": "user", "content": prompt},
                ]
    )
    output_text = response["choices"][0]["message"]["content"]

    return output_text

print(gpt_call("What is the meaning of life?"))
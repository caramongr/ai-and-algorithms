# imports

import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
openai = OpenAI()
MODEL = 'gpt-4o-mini'


system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales evemt.'\
Encourage the customer to buy hats if they are unsure what to get."


def chat(message, history):
    messages = [{"role": "system", "content": system_message}]
    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})

    if 'belt' in message:
        messages.append({"role": "system", "content": "For added context, the store does not sell belts, \
but be sure to point out other items on sale"})
    
    messages.append({"role": "user", "content": message})

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response



gr.ChatInterface(fn=chat).launch()


# imports

import io
import os
import subprocess
import sys

# import anthropic
# import google.generativeai
import gradio as gr
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from openai import OpenAI

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
# os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')

# initialize
# NOTE - option to use ultra-low cost models by uncommenting last 2 lines

openai = OpenAI()
# claude = anthropic.Anthropic()
OPENAI_MODEL = "gpt-4o"
# CLAUDE_MODEL = "claude-3-5-sonnet-20240620"

# Want to keep costs ultra-low? Uncomment these lines:
# OPENAI_MODEL = "gpt-4o-mini"
# CLAUDE_MODEL = "claude-3-haiku-20240307"


system_message = "You are an assistant that reimplements Python code in high performance C++ for an M1 Mac. "
system_message += "Respond only with C++ code; use comments sparingly and do not provide any explanation other than occasional comments. "
system_message += "The C++ response needs to produce an identical output in the fastest possible time."


def user_prompt_for(python):
    user_prompt = "Rewrite this Python code in C++ with the fastest possible implementation that produces identical output in the least time. "
    user_prompt += "Respond only with C++ code; do not explain your work other than a few comments. "
    user_prompt += "Pay attention to number types to ensure no int overflows. Remember to #include all necessary C++ packages such as iomanip.\n\n"
    user_prompt += python
    return user_prompt

def messages_for(python):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt_for(python)}
    ]

# write to a file called optimized.cpp

def write_output(cpp):
    code = cpp.replace("```cpp","").replace("```","")
    with open("optimized.cpp", "w") as f:
        f.write(code)

def optimize_gpt(python):    
    stream = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages_for(python), stream=True)
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        print(fragment, end='', flush=True)
    write_output(reply)

pi = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(100_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""

exec(pi)
optimize_gpt(pi)
exec(pi)
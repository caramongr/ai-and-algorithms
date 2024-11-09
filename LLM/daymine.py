# imports

import http.client
import json
import os
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# Initialization

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
MODEL = "gpt-4o-mini"
openai = OpenAI()

system_message = "You are a helpful assistant that can help customers with the weather. You can also tell the time and date"
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."

def chat(message, history):
    messages = [{"role": "system", "content": system_message}]
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    print(response)

    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        tool_response = handle_tool_call(message)
        print("---------------")
        print(tool_response)
        print("---------------")
        messages.append(message)
        messages.append(tool_response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
    
    return response.choices[0].message.content

# Let's start by making a useful function

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")

def get_current_day_date_time(request_type):
    now = datetime.now()
    
    if request_type == "day":
        current_day = now.strftime("%A")  # E.g., Monday
        response = f"Today is {current_day}."
    elif request_type == "date":
        current_date = now.strftime("%Y-%m-%d")  # E.g., 2024-10-29
        response = f"The current date is {current_date}."
    elif request_type == "time":
        current_time = now.strftime("%H:%M:%S")  # E.g., 14:23:05
        response = f"The current time is {current_time}."
    elif request_type == "day_date":
        current_day = now.strftime("%A")
        current_date = now.strftime("%Y-%m-%d")
        response = f"Today is {current_day}, {current_date}."
    elif request_type == "date_time":
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        response = f"The current date and time is {current_date}, {current_time}."
    elif request_type == "day_time":
        current_day = now.strftime("%A")
        current_time = now.strftime("%H:%M:%S")
        response = f"It is {current_day}, {current_time}."
    elif request_type == "all":
        current_day = now.strftime("%A")
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        response = f"Today is {current_day}, {current_date}, and the current time is {current_time}."
    else:
        response = "I'm sorry, I couldn't understand your request."
    
    return response

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}


# Tool function metadata for getting current day, date, and time
day_date_time_function = {
    "name": "get_current_day_date_time",
    "description": "Get the current day, date, or time from the system.",
    "parameters": {
        "type": "object",
        "properties": {
            "request_type": {
                "type": "string",
                "description": (
                    "The type of information requested: 'day', 'date', 'time', "
                    "'day_date', 'date_time', 'day_time', or 'all'."
                )
            }
        },
        "required": ["request_type"],
        "additionalProperties": False
    }
}




def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    # print(f"Tool call: {tool_call.function.name}")
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    print(f"Arguments: {arguments}")
        
    if function_name == "get_current_day_date_time":
        request_type = arguments.get("request_type")
        response_content = get_current_day_date_time(request_type)
        return {
            "role": "tool",
            # "content": response_content,
            "content": json.dumps({"request_type": response_content}),
            "tool_call_id": tool_call.id
        }

# tools = [{"type": "function", "function": price_function}]
tools = [{"type": "function", "function": day_date_time_function}]

gr.ChatInterface(fn=chat).launch()
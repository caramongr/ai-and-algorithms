# Imports
# doesnt work onlz for second parameter example

import http.client
import json
import os
from datetime import datetime

import gradio as gr
import pytz
from dotenv import load_dotenv
from openai import OpenAI
from timezonefinder import TimezoneFinder

# Initialization

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-openai-api-key')
MODEL = "gpt-3.5-turbo"  # Use a valid model name
openai = OpenAI()

system_message = (
    "You are a helpful assistant that can help customers with the weather. "
    "You can also tell the time and date. "
    "If the user asks for the time without specifying a location, politely ask for their city. "
    "Give short, courteous answers, no more than 1 sentence. "
    "Always be accurate. If you don't know the answer, say so."
)

# Function to get current day, date, and time with location support
def get_current_day_date_time(request_type, location=None):
    # Use TimezoneFinder to get the time zone from the location
    if location:
        tf = TimezoneFinder()
        # For simplicity, we'll use a predefined mapping of city names to coordinates
        city_coordinates = {
            'athens': (37.9838, 23.7275),
            'london': (51.5074, -0.1278),
            'new york': (40.7128, -74.0060),
            'tokyo': (35.6895, 139.6917),
            'paris': (48.8566, 2.3522),
            'sydney': (-33.8688, 151.2093),
            'berlin': (52.5200, 13.4050),
            # Add more cities as needed
        }
        city = location.lower()
        if city in city_coordinates:
            lat, lng = city_coordinates[city]
            timezone_str = tf.timezone_at(lat=lat, lng=lng)
            if timezone_str is None:
                timezone_str = 'UTC'
        else:
            timezone_str = 'UTC'  # Default to UTC if city not found
    else:
        timezone_str = 'UTC'  # Default to UTC if no location provided

    # Get the current time in the specified time zone
    timezone = pytz.timezone(timezone_str)
    now = datetime.now(timezone)

    # Format the date and time based on request_type
    if request_type == "day":
        current_day = now.strftime("%A")
        response = f"Today is {current_day} in {location.title()}."
    elif request_type == "date":
        current_date = now.strftime("%Y-%m-%d")
        response = f"The current date in {location.title()} is {current_date}."
    elif request_type == "time":
        current_time = now.strftime("%H:%M:%S")
        response = f"The current time in {location.title()} is {current_time}."
    elif request_type == "day_date":
        current_day = now.strftime("%A")
        current_date = now.strftime("%Y-%m-%d")
        response = f"Today is {current_day}, {current_date} in {location.title()}."
    elif request_type == "date_time":
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        response = f"The current date and time in {location.title()} is {current_date}, {current_time}."
    elif request_type == "day_time":
        current_day = now.strftime("%A")
        current_time = now.strftime("%H:%M:%S")
        response = f"It is {current_day}, {current_time} in {location.title()}."
    elif request_type == "all":
        current_day = now.strftime("%A")
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        response = f"Today is {current_day}, {current_date}, and the current time is {current_time} in {location.title()}."
    else:
        response = "I'm sorry, I couldn't understand your request."
    
    return response

# Tool function metadata for getting current day, date, and time
day_date_time_function = {
    "name": "get_current_day_date_time",
    "description": "Get the current day, date, or time from the system, possibly in a specified location.",
    "parameters": {
        "type": "object",
        "properties": {
            "request_type": {
                "type": "string",
                "description": (
                    "The type of information requested: 'day', 'date', 'time', "
                    "'day_date', 'date_time', 'day_time', or 'all'."
                )
            },
            "location": {
                "type": "string",
                "description": "The city name to get the time for (e.g., 'London', 'Tokyo')."
            }
        },
        "required": ["request_type"],
        "additionalProperties": False
    }
}

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    print(f"Arguments: {arguments}")
        
    if function_name == "get_current_day_date_time":
        request_type = arguments.get("request_type")
        location = arguments.get("location")
        if not location:
            return {
                "role": "assistant",
                "content": "Could you please specify your city so I can provide the local time?"
            }
        response_content = get_current_day_date_time(request_type, location)
        return {
            "role": "assistant",
            "content": response_content
        }

def chat(messages):
    conversation = [{"role": "system", "content": system_message}] + messages

    # Ensure messages have valid content
    for msg in conversation:
        if msg['content'] is None:
            msg['content'] = "No content provided."

    response = openai.chat.completions.create(model=MODEL, messages=conversation, tools=tools)
    print(f"OpenAI Response: {response}")

    if response.choices[0].finish_reason == "tool_calls":
        message_response = response.choices[0].message
        tool_response = handle_tool_call(message_response)
        print("---------------")
        print(tool_response)
        print("---------------")
        # Append the assistant's message triggering the tool call
        conversation.append({"role": "assistant", "content": message_response.content})
        # Append the tool's response as an assistant message
        conversation.append({"role": "assistant", "content": tool_response['content']})
        response = openai.chat.completions.create(model=MODEL, messages=conversation)

    assistant_reply = response.choices[0].message.content
    return assistant_reply

tools = [{"type": "function", "function": day_date_time_function}]

# Launch Gradio chat interface
gr.ChatInterface(fn=chat, chatbot=gr.Chatbot(type="messages")).launch()

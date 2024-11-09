import os

import openai
from dotenv import load_dotenv

# Set your API key
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
openai.api_key = os.environ['OPENAI_API_KEY']

def get_completion(prompt, model="gpt-3.5-turbo"):
    """
    Get a completion from the OpenAI API.
    
    Args:
        prompt (str): The prompt to send to the API
        model (str): The model to use (default: gpt-3.5-turbo)
        
    Returns:
        str: The generated completion
    """
    try:
        # Create the messages array with the system and user messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # Make the API call
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=150,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Extract and return the generated text
        return response.choices[0].message.content
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Example prompts
    prompts = [
        "What is the capital of France?",
        "Write a short poem about programming"
    ]
    
    # Test the function with different prompts
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Response:", get_completion(prompt))
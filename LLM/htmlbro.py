# imports
import json
import os

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables and initialize OpenAI API key
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
MODEL = 'gpt-4o-mini'
openai = OpenAI()

# A class to represent a Webpage
class Website:
    url: str
    title: str
    text: str

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""

    def get_summary_prompt(self):
        return f"Website Title: {self.title}\nWebsite Content:\n{self.text[:1500]}..."

def clean_response(response_text):
    # Remove code block markers (```json and ```)
    cleaned_text = response_text.replace('```json', '').replace('```', '').strip()
    return cleaned_text

def summarize_content_to_json(content):
    # Send the content to OpenAI for summarization and request a JSON response
    completion = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Summarize the following website content into key sections with relevant titles. Respond in JSON format with section titles and content. Ensure the summary does not exceed 2000 words."},
            {"role": "user", "content": content}
        ]
    )

    # Extract the response from the OpenAI API
    response_text = completion.choices[0].message.content.strip()

    # Clean the response to ensure it is valid JSON
    cleaned_response = clean_response(response_text)

    # Try to parse the cleaned response as JSON
    try:
        json_response = json.loads(cleaned_response)
    except json.JSONDecodeError:
        print("Error: Response is not in valid JSON format after cleaning.")
        print("Raw Response:", cleaned_response)
        return None

    return json_response

def create_html_from_json(json_data):
    if not json_data:
        return "<html><body><h1>Error creating brochure</h1><p>Unable to parse JSON data.</p></body></html>"
    print("-----------")
    print(json_data)
    print("-----------")
    # Determine the title for the HTML brochure
    title = json_data.get('website_title', 'Company Brochure')

    # Generate HTML content using the JSON data
    html_content = f"""
    <html>
    <head>
        <title>Brochure for {title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            p {{ line-height: 1.6; }}
            .section {{ margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Brochure for {title}</h1>
    """

    # Dynamically loop through the JSON data to generate sections
    for section_title, section_content in json_data.items():
        if isinstance(section_content, dict):
            section_details = ' '.join([f'{key}: {value}' for key, value in section_content.items()])
        else:
            section_details = section_content
        html_content += f"""
        <div class="section">
            <h2>{section_title}</h2>
            <p>{section_details}</p>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    return html_content

def create_brochure(url):
    website = Website(url)
    print(f"Website Title: {website.title}")

    json_summary = summarize_content_to_json(website.get_summary_prompt())

    # if json_summary:
    #     # print("JSON Summary:")
    #     # print(json_summary)
    # else:
    #     print("Failed to create JSON summary.")

    # Convert the JSON response into an HTML brochure
    html_brochure = create_html_from_json(json_summary)

    return html_brochure

# Example usage
url = "https://wizdom.gr"
html_brochure = create_brochure(url)

# Saving the HTML brochure to a file
with open("brochure.html", "w", encoding="utf-8") as file:
    file.write(html_brochure)

print("Brochure saved as brochure.html")

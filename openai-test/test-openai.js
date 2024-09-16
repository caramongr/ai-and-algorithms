require('dotenv').config();
const axios = require('axios');

const apiKey = process.env.OPENAI_API_KEY;

const prompt = "Tell me a joke.";


const data = {
  model: "gpt-3.5-turbo", // Updated model
  messages: [{ role: "user", content: prompt }],
  max_tokens: 50
};

const headers = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${apiKey}`
};

axios.post('https://api.openai.com/v1/chat/completions', data, { headers: headers })
  .then(response => {
    console.log("Response from OpenAI API:");
    console.log(response.data.choices[0].message.content.trim());
  })
  .catch(error => {
    console.error("Error calling OpenAI API:");
    console.error(error.response ? error.response.data : error.message);
  });
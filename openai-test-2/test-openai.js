const axios = require('axios');
const express = require('express');
const app = express();
const path = require('path');
const bodyParser = require('body-parser');

// Replace 'test-key' with your actual OpenAI API key
const apiKey = process.env.OPENAI_API_KEY;

app.use(bodyParser.json());
app.use(express.static('public'));

const generatePhysicsLabTest = async (topic, numQuestions, language) => {
  const prompt = `Create a lab test for a physics class on the topic of ${topic}. Include ${numQuestions} detailed questions and their expected answers. The test should be in ${language}. Format the output as a JSON array with each element containing a 'question' and an 'answer' field.`;

  const data = {
    model: "gpt-3.5-turbo",
    messages: [{ role: "user", content: prompt }],
    max_tokens: 500
  };

  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`
  };

  try {
    const response = await axios.post('https://api.openai.com/v1/chat/completions', data, { headers: headers });
    const labTest = JSON.parse(response.data.choices[0].message.content.trim());
    return labTest;
  } catch (error) {
    console.error("Error calling OpenAI API:");
    console.error(error.response ? error.response.data : error.message);
    return null;
  }
};

const evaluateResponses = async (responses, language) => {
  const results = await Promise.all(responses.map(async response => {
    const prompt = `Evaluate the following student's answer to a question in ${language}. Determine if the answer is correct, partially correct, or incorrect, and explain why.\n\nQuestion: ${response.question}\n\nExpected Answer: ${response.expectedAnswer}\n\nStudent's Answer: ${response.studentAnswer}\n\nEvaluation:`;

    const data = {
      model: "gpt-3.5-turbo",
      messages: [{ role: "user", content: prompt }],
      max_tokens: 150
    };

    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    };

    try {
      const evaluationResponse = await axios.post('https://api.openai.com/v1/chat/completions', data, { headers: headers });
      const evaluationText = evaluationResponse.data.choices[0].message.content.trim();
      const isCorrect = evaluationText.toLowerCase().includes('correct') && !evaluationText.toLowerCase().includes('incorrect');
      return { ...response, isCorrect, evaluation: evaluationText };
    } catch (error) {
      console.error("Error calling OpenAI API for evaluation:");
      console.error(error.response ? error.response.data : error.message);
      return { ...response, isCorrect: false, evaluation: 'Error evaluating the answer.' };
    }
  }));

  return results;
};

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post('/generate-test', async (req, res) => {
  const { topic, numQuestions, language } = req.body;
  const questions = await generatePhysicsLabTest(topic, numQuestions, language);
  if (!questions) {
    return res.status(500).send("Failed to generate questions.");
  }
  res.json(questions);
});

app.post('/submit-responses', async (req, res) => {
  const { responses, language } = req.body;
  const results = await evaluateResponses(responses, language);
  res.json(results);
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});

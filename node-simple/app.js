const express = require('express');
const cors = require('cors');
const app = express();
const PORT = 3000;

// Enable CORS for all routes
app.use(cors());

// Middleware to parse JSON and URL-encoded bodies
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// GET endpoint at '/'
app.get('/', (req, res) => {
    res.status(200).send('It works!\n');
});

// GET endpoint at '/new-endpoint'
app.get('/new-endpoint', (req, res) => {
    res.status(200).send('You have reached the new endpoint!\n');
});

// POST endpoint at '/post-endpoint'
app.post('/post-endpoint', (req, res) => {
    res.status(200).json({ received: req.body });
});

// 404 handler for unknown routes
app.use((req, res) => {
    res.status(404).send('404 Not Found\n');
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is listening on port ${PORT}`);
});

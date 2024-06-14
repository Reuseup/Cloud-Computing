const express = require('express');
const { getPythonResponse } = require('./services/pythonService');

const router = express.Router();

// Define your API endpoint
router.post('/process', async (req, res, next) => {
  try {
    const userInput = req.body.inputText; // Assuming input text is in `inputText` field
    if (!userInput) {
      return res.status(400).json({ error: 'Input text is required' });
    }
    
    const response = await getPythonResponse(userInput);
    res.json({ response });
  } catch (err) {
    next(err);
  }
});

module.exports = router;

const axios = require('axios');

// Function to call the Python service and get the response
const getPythonResponse = async (inputText) => {
  try {
    // Replace with your actual Python service URL
    const pythonServiceUrl = 'http://your-python-service-url/process';

    const response = await axios.post(pythonServiceUrl, { inputText });

    return response.data;
  } catch (err) {
    throw new Error(`Error while calling Python service: ${err.message}`);
  }
};

module.exports = {
  getPythonResponse,
};

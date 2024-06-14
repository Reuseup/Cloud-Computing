const express = require('express');
const { handleError } = require('./handlers/exceptionHandler');
const routes = require('./routes');

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json()); // To parse JSON request bodies

// Use the routes defined in routes.js
app.use('/', routes);

// Error handling middleware
app.use((err, req, res, next) => {
  handleError(err, res);
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

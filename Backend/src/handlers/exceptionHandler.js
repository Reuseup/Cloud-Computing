const handleError = (err, res) => {
    res.status(err.statusCode || 500).json({
      error: {
        message: err.message || 'An unknown error occurred',
      },
    });
  };
  
  module.exports = {
    handleError,
  };
  
// response-generator.js

const fs = require('fs');
const csv = require('csv-parser');
const { AutoTokenizer, AutoModelForSequenceClassification } = require('@xenova/transformers');
const tf = require('@tensorflow/tfjs-node');

// Load the CSV file and parse it
async function loadDataset() {
    const questions = [];
    const answers = [];

    return new Promise((resolve, reject) => {
        fs.createReadStream('Conversation-cleaned.csv')
            .pipe(csv())
            .on('data', (row) => {
                questions.push(row.question);
                answers.push(row.answer);
            })
            .on('end', () => {
                resolve({ questions, answers });
            })
            .on('error', reject);
    });
}

async function main() {
    const numLayersToUse = 1; // Number of layers to use for embedding

    // Load the tokenizer and model
    const tokenizer = await AutoTokenizer.from_pretrained('indobenchmark/indobert-lite-base-p1');
    const model = await AutoModelForSequenceClassification.from_pretrained('indobenchmark/indobert-lite-base-p1');

    // Load the dataset
    const { questions, answers } = await loadDataset();

    // Function to generate response
    async function generateResponse(inputText) {
        // Tokenize and encode input text
        const inputTokens = tokenizer.encode(inputText, {
            returnTensors: 'tf',
            padding: true,
            truncation: true,
        });

        // Get BERT model embeddings for input text
        const inputEmbeddings = await getEmbeddings(inputTokens);

        // Find the most similar predefined response based on embeddings
        let maxSimilarity = -1;
        let bestResponse = "";

        for (let i = 0; i < questions.length; i++) {
            const questionTokens = tokenizer.encode(questions[i], {
                returnTensors: 'tf',
                padding: true,
                truncation: true,
            });

            const questionEmbeddings = await getEmbeddings(questionTokens);

            // Compute cosine similarity
            const similarity = computeCosineSimilarity(inputEmbeddings, questionEmbeddings);

            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                bestResponse = answers[i];
            }
        }

        return bestResponse;
    }

    // Function to compute embeddings using the model
    async function getEmbeddings(tokens) {
        const outputs = await model(tokens, { output_hidden_states: true });
        const hiddenStates = outputs.hidden_states.slice(-numLayersToUse);
        const stackedStates = tf.stack(hiddenStates);
        const meanState = stackedStates.mean(0).mean(1);
        return meanState;
    }

    // Function to compute cosine similarity
    function computeCosineSimilarity(embedding1, embedding2) {
        const dotProduct = tf.sum(tf.mul(embedding1, embedding2));
        const norm1 = tf.norm(embedding1);
        const norm2 = tf.norm(embedding2);
        return dotProduct.div(tf.mul(norm1, norm2)).arraySync();
    }

    // Example usage
    const userInput = "Apa kabar?";
    const response = await generateResponse(userInput);
    console.log(`Response: ${response}`);
}

main().catch(console.error);

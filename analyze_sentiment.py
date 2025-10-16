from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Load Pretrained Model and Tokenizer ---
# Using the CardiffNLP RoBERTa model fine-tuned for sentiment analysis on Twitter data
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Tokenizer converts text into model-readable tokens
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Model performs sequence classification (e.g., Positive / Negative / Neutral)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Define sentiment labels corresponding to model output indices
labels = ['Negative', 'Neutral', 'Positive']


# --- Function: Analyze Sentiment of a Single Text ---
def analyze_sentiment(text):
    """
    Analyze the sentiment of a single comment or paragraph.
    Returns:
        - result: predicted sentiment label (Positive/Negative/Neutral)
        - confidence: model's confidence score for the prediction
    """
    # Tokenize input text into model-friendly format (convert text → tensor)
    inputs = tokenizer(
        text,
        return_tensors="pt",  # Return PyTorch tensors
        truncation=True,      # Truncate long text to fit model input size
        padding=True          # Add padding if needed
    )

    # Run text through the model to get raw prediction scores (logits)
    outputs = model(**inputs)

    # Apply softmax to convert logits into probability distribution
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Find the sentiment label with the highest probability
    result = labels[torch.argmax(probs)]

    # Extract the confidence (probability) of the predicted label
    confidence = torch.max(probs).item()

    return result, confidence


# --- Function: Analyze Multiple Comments at Once ---
def analyze_multiple_comments(comments):
    """
    Analyze a list of comments or paragraphs.
    Returns:
        - counts: dictionary with the total count of each sentiment
        - results: detailed list of individual predictions and confidence scores
    """
    # Initialize counters for summary statistics
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

    # Store detailed sentiment results for each comment
    results = []

    # Loop through each comment in the list
    for comment in comments:
        comment = comment.strip()
        if not comment:
            continue  # Skip empty lines

        # Analyze sentiment of the individual comment
        sentiment, confidence = analyze_sentiment(comment)

        # Update summary count
        counts[sentiment] += 1

        # Store detailed results for display
        results.append({
            "comment": comment,
            "sentiment": sentiment,
            "confidence": confidence
        })

    # Return both summary counts and detailed breakdown
    return counts, results

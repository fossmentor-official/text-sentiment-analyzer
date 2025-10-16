import streamlit as st  # Streamlit for creating the web app interface
from analyze_sentiment import analyze_multiple_comments  # Custom function for multi-comment sentiment analysis

# --- Apply Custom CSS Styling to Make All Text White ---
st.markdown(
    """
    <style>
        /* Set overall text color to white and background to dark mode */
        html, body, [class*="css"], section {
            color: white !important;
            background-color: #0E1117;  /* dark theme for better contrast */
        }

        /* Style text areas, labels, and paragraphs */
        .stTextArea textarea, label, p {
            background-color: #1E1E1E;
            color: white;
        }

        /* Customize Analyze button appearance */
        .stButton button {
            background-color: #262730;
            color: white;
            border-radius: 8px;
            border: 1px solid #4CAF50;  /* subtle green border */
        }

        /* Add hover effect for button */
        .stButton button:hover {
            background-color: #4CAF50;
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True  # Allows HTML/CSS injection for styling
)

# --- App Title and Description ---
st.title("🧠 Text Sentiment Analyzer")
st.write("Analyze the emotional tone of multiple comments or paragraphs using Transformers (BERT/Roberta)!")

# --- Input Section for User Comments ---
text = st.text_area("Enter your comments (one per line):")

# --- Action Button to Trigger Sentiment Analysis ---
if st.button("Analyze"):
    # Check if text input is not empty
    if text.strip():
        # Split text into individual lines, ignoring empty ones
        comments = [line for line in text.split("\n") if line.strip()]

        # Call the sentiment analysis function on all comments
        summary_counts, detailed_results = analyze_multiple_comments(comments)

        # --- Display Summary of Sentiments ---
        st.subheader("✅ Summary Counts")
        st.write(summary_counts)  # Example: {'positive': 5, 'negative': 2, 'neutral': 3}

        # --- Display Detailed Results for Each Comment ---
        st.subheader("📄 Detailed Results")
        for r in detailed_results:
            st.write(f"**Comment:** {r['comment']}")  # Original comment
            st.write(f"**Sentiment:** {r['sentiment']} | **Confidence:** {r['confidence']:.2f}")  # Prediction and score
            st.write("---")  # Separator line for clarity
    else:
        # Warn user if no text was entered
        st.warning("Please enter some text to analyze!")

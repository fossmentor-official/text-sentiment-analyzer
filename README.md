# 🧠 Customer Feedback Sentiment Analyzer — Real-Time Opinion Intelligence for eCommerce & SaaS

> **Instantly classify customer reviews, support tickets, and social comments as Positive, Neutral, or Negative — at scale, without expensive GPU infrastructure.**

eCommerce businesses and SaaS platforms collect thousands of customer feedback entries every month — but most teams only read a fraction of them. Negative sentiment builds up undetected until it becomes a churn or reputation problem.

This tool solves that. Powered by the **CardiffNLP Twitter RoBERTa** transformer model and a clean **Streamlit** interface, it processes multiple inputs simultaneously and surfaces sentiment distribution in seconds — not hours.

---

## 💡 Business Use Cases

- **eCommerce** — Analyze product reviews in bulk to catch negative sentiment before it impacts ratings
- **SaaS Platforms** — Triage support tickets by emotional tone to prioritize critical customer issues
- **Marketing Teams** — Monitor social media comments for brand sentiment without manual reading
- **Customer Success** — Identify at-risk customers based on negative feedback patterns

---

## 🚀 Features

- Real-time sentiment classification using HuggingFace Transformers (RoBERTa)
- Processes multiple paragraphs or comments in a single run
- Displays sentiment distribution — Positive / Neutral / Negative counts
- Clean, non-technical Streamlit UI (usable by business teams, not just developers)
- **CPU-compatible** — No GPU or expensive cloud inference required
- Fully containerized with Docker for plug-and-play deployment on AWS, GCP, or DigitalOcean

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| AI Model | HuggingFace Transformers — CardiffNLP RoBERTa |
| Backend | Python |
| Frontend | Streamlit |
| Containerization | Docker / Dev Containers |
| Deployment | Any cloud (CPU-based inference) |


## Steps Installation

1. Clone the repository:
```bash
git clone https://github.com/fossmentor-official/text-sentiment-analyzer.git
cd text-sentiment-analyzer
```

2. Open in VS Code / Cursor and click:
“Reopen in Container”

3. Wait for Docker to build the image. Once ready, the environment will open inside the container.

4. Run the Streamlit app:
```bash
streamlit run app.py
```

5. Open the URL shown in the terminal usually http://localhost:8501


---

## 🖥️ Usage

1. Paste one or more paragraphs, reviews, or comments into the text box
2. Click **"Analyze"**
3. The app returns:
   - Individual sentiment label per paragraph
   - Total distribution count — Positive / Neutral / Negative

---

## 🗺️ Roadmap

- [ ] REST API endpoint for integration with external platforms (Shopify, Zendesk, HubSpot)
- [ ] Live feed integration — Facebook, Twitter/X, Google Reviews APIs
- [ ] Visual sentiment trend charts over time
- [ ] Multi-language support
- [ ] CSV/database export for reporting pipelines
- [ ] Webhook support for real-time alerting on negative sentiment spikes

---

## 🤝 Built by Wasif Younas

**AI Automation & Solutions Architect** with 12+ years of experience building production-grade systems across healthcare SaaS, eCommerce, and ERP platforms.

This project is part of a broader focus on helping businesses integrate AI into real operational workflows — not just demos.

**Looking to implement something similar for your business?**

[![LinkedIn](https://raw.githubusercontent.com/MartinHeinz/MartinHeinz/master/linkedin-3-16.png)](https://www.linkedin.com/in/fossmentor/)
[![Facebook](https://i.imgur.com/dqSkGWu.png)](https://facebook.com/fossmentor)
[![Instagram](https://i.imgur.com/TFy6wii.png)](https://www.instagram.com/fossmentor.official/)

📧 contact@fossmentor.com | 🌐 [fossmentor.com](https://fossmentor.com)

---

## 🪪 License

MIT License © 2026 Fossmentor Official — [fossmentor.com](https://fossmentor.com)
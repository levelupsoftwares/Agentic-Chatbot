# 💬 RAG Chatbot (Retrieval-Augmented Generation)

An intelligent chatbot built using **LangChain + Groq LLM + Hybrid Search** that retrieves relevant information and generates accurate, context-aware responses.

---

## 📸 Project Preview

<!-- 🔽 Replace the path below with your image -->
![App Screenshot](.//assets/preview_Img.png)



---

## 🚀 Features

- 🔍 **Hybrid Retrieval System**
  - Combines semantic + keyword-based search
- 🧠 **Query Rewriting**
  - Converts vague queries into clear standalone questions using LLM
- 💬 **Chat History Awareness**
  - Maintains context for better conversation flow
- ⚡ **Fast Responses**
  - Powered by Groq LLM
- 🧾 **Context Injection**
  - Uses retrieved documents to improve answer quality
- 🌐 **Streamlit UI**
  - ChatGPT-like interface with typing effect
- 🧹 **Clear Chat Option**
  - Reset conversation anytime

---




## ⚙️ Installation

### 1. Clone the repository
```bash
git clone <https://github.com/levelupsoftwares/Agentic-Chatbot>
cd chatbot
```
---

### 2. Create virtual environment

``` bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```
---

## 🔑 Environment Variables

Create a .env file:
```
GROQ_API_KEY='your-API-KEY-HERE'
HUGGINGFACEHUB_API_TOKEN='Your-HUGGINGFACE-API-KEY-HERE'

```

---
## ▶️ Run the Project
🌐 Run UI (Streamlit)
```
streamlit run main.py
```
---

### 💻 Run Terminal Version (Optional)
```
python3 src/chains/rag_chain.py
```
----

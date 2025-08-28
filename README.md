# The Transcript Intelligence  

The Transcript Intelligence is a **Streamlit-powered web application** that lets you ask questions about any YouTube video with English captions. By leveraging **LangChain**, **FAISS**, **HuggingFace embeddings**, and **Perplexity API**, it retrieves answers **only from the video transcript**, saving you from watching long videos.  

---

## 🚀 Features  
- 🔗 Paste a YouTube video URL  
- 📑 Fetches and processes the transcript automatically (English captions only)  
- ❓ Ask natural language questions about the video  
- ⚡ Uses **RAG (Retrieval-Augmented Generation)** for accurate responses  
- 🎨 Modern dark-themed UI with animations  

---

## 🛠️ Tech Stack  
- **Python**  
- **Streamlit** (UI framework)  
- **YouTube Transcript API** (to fetch captions)  
- **LangChain** + **FAISS** (for semantic search & retrieval)  
- **HuggingFace Embeddings**  
- **ChatPerplexity (LLM)**  

---

## ⚙️ Setup Instructions  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/transcript-intelligence.git
cd transcript-intelligence
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Add API Keys
**Create a .env file in the project root and add your keys:**
```bash
# Perplexity API Key
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# HuggingFace API Key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key_here
```

### 5️⃣ Run the app
```bash
streamlit run Youtube_transcript_project.py
```




import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_perplexity import ChatPerplexity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

# Helper Functions
def id_extractor(url):
    try:
        return url.split('v=')[1].split('&')[0]
    except IndexError:
        return None

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

@st.cache_resource(show_spinner=False)
def load_model_and_embeddings():
    model = ChatPerplexity()
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return model, embedding

def generate_chain(transcript, model, embedding):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 4})
    
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        Don't mention in your response like 'from transcript context or [transcript context] and dont add any citation at last'.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | model | parser
    return main_chain

# Streamlit UI
st.set_page_config(page_title="The Transcript Intelligence", page_icon="RAG\Youtube_Chatbot\icons8-youtube-120.png",layout="wide")

# Inject custom dark CSS
st.markdown("""
    
    <style>
    body {
        background-color: #0d0d0d;
        color: #e6e6e6;
    }
    .main {
        background-color: #0d0d0d;
        color: #e6e6e6;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        font-family: 'Segoe UI', sans-serif;
        font-size: 3.5em;
        font-weight: 700;
        color: #ffffff;
        text-align: left;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); /* soft shadow for readability */
        letter-spacing: 1px;
        animation: fadeInDown 1.5s ease-out;
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

       

    p {
        font-size: 1.2em;
        color: #cccccc;
    }
    
    input[type="text"],
    textarea {
        background-color: #1a1a1a !important;
        color: #f2f2f2 !important;
        border: 2px solid #444 !important;
        border-radius: 10px !important;
        padding: 0.7em 1em !important;
        font-size: 1.05em !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease, background-color 0.25s ease;
        box-shadow: 0 0 4px rgba(255, 51, 51, 0.15);
    }

/* On hover */
    input[type="text"]:hover,
    textarea:hover {
        border-color: #ff4d4d !important;
        box-shadow: 0 0 8px rgba(255, 77, 77, 0.4);
    }

    /* On focus */
    input[type="text"]:focus,
    textarea:focus {
        border-color: #ff1a1a !important;
        box-shadow: 0 0 10px rgba(255, 26, 26, 0.6);
        outline: none !important;
        background-color: #262626 !important;
    }
    div.stButton > button {
        background-color: #b30000;  /* dark red */
        color: white;
        padding: 0.7em 1.5em;
        font-size: 16px;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    div.stButton > button:hover {
        background-color: #ff1a1a;  /* brighter red on hover */
        color: black;
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(255, 26, 26, 0.5);
    }
    .logo {
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: linear-gradient(135deg, #ff3333, #990000);
        animation: rotate 6s linear infinite;
        margin: auto;
        box-shadow: 0 0 20px #ff3333;
    }
   
    .stApp {
        background-image: url("https://t4.ftcdn.net/jpg/01/63/61/53/360_F_163615394_d8VNpuO5Lv7TtoJxmXzuiLJVHAVbER1h.jpg");
        background-attachment: fixed;
        background-size: cover;
    }

     @keyframes fadeInAnswer {
            0% {
                opacity: 0;
                transform: translateY(10px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }        

    @keyframes rotate {
        from {transform: rotate(0deg);}
        to {transform: rotate(360deg);}
    }
    </style>
""", unsafe_allow_html=True)

hide_menu = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

# Page layout
left, right = st.columns(2)

with left:
    st.markdown("<h1>The Transcript Intelligence</h1>", unsafe_allow_html=True)
    st.markdown('<p>Ask questions about any YouTube video (with English captions).</p>', unsafe_allow_html=True)


    url = st.text_input("Enter YouTube Video URL")
    question = st.text_area("Ask your question")
    if st.button("Get Answer"):
        if url.strip() == "" or question.strip() == "":
            st.warning("Please enter both the URL and your question.")
        else:
            video_id = id_extractor(url)

            if video_id is None:
                st.error("Invalid YouTube URL format.")
            else:
                try:
                    with st.spinner("Fetching and analyzing transcript..."):
                        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
                        transcript_text = " ".join([entry.text for entry in transcript_list])

                        model, embedding = load_model_and_embeddings()
                        chain = generate_chain(transcript_text, model, embedding)

                        answer = chain.invoke(question)
                    
                        st.markdown("""
                                <div style="
                                    background: linear-gradient(145deg, #1a1a1a, #262626);
                                    border: 1px solid #ff4d4d;
                                    padding: 20px;
                                    border-radius: 15px;
                                    margin-top: 20px;
                                    box-shadow: 0 0 20px rgba(255, 77, 77, 0.4);
                                    color: #f2f2f2;
                                    font-size: 1.1em;
                                    line-height: 1.6;
                                    animation: fadeInAnswer 0.8s ease-in-out;
                                    font-family: 'Segoe UI', sans-serif;
                                    transition: all 0.3s ease-in-out;
                                    ">
                                    <strong>Answer:</strong><br><br>{}
                                </div>
                            """.format(answer), unsafe_allow_html=True)


                except (TranscriptsDisabled, NoTranscriptFound):
                    st.error("Transcript is not available for this video.")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")




with right:
    st.markdown("""
    <div class="yt-circle-wrapper">
        <div class="yt-circle">
            <img src="https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png" class="yt-logo" alt="YouTube Logo">
        </div>
    </div>

    <style>
    .yt-circle-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 420px;
    }

    .yt-circle {
        width: 320px;
        height: 320px;
        border-radius: 50%;
        background: radial-gradient(circle, #ff3333, #990000);
        box-shadow: 0 0 40px #ff3333aa, 0 0 80px #990000aa;
        display: flex;
        justify-content: center;
        align-items: center;
        animation: rotate 12s linear infinite, pulse 3s ease-in-out infinite;
        transition: all 0.3s ease-in-out;
    }

    .yt-logo {
        width: 130px;
        filter: drop-shadow(0 0 12px #ffffffcc);
        transition: transform 0.3s ease-in-out;
    }

    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 40px #ff3333aa, 0 0 80px #990000aa; }
        50% { transform: scale(1.05); box-shadow: 0 0 60px #ff3333ee, 0 0 100px #990000ee; }
        100% { transform: scale(1); box-shadow: 0 0 40px #ff3333aa, 0 0 80px #990000aa; }
    }
    </style>
    """, unsafe_allow_html=True)

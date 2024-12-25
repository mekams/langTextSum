# Import necessary libraries
import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from rouge_score import rouge_scorer
from docx import Document
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Preprocessing: Splitting text using LangChain's CharacterTextSplitter
def preprocess_text(text, chunk_size=7000, chunk_overlap=2500):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# Extractive Summarization: LexRank implementation
def extractive_summarization(text, num_sentences=20):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join([str(sentence) for sentence in summary])

# Abstractive Summarization using LangChain (Facebook BART)
prompt = PromptTemplate(
    input_variables=["text"],
    template="Provide a detailed and comprehensive summary of the following text: {text}"
)

# Hugging Face BART (Parameters)
llm = HuggingFaceHub(
    repo_id="facebook/bart-large-cnn",
    task="summarization",
    huggingfacehub_api_token=st.secrets["huggingface_token"],  # private token : Kamran Mustafa (Hugging face)
    model_kwargs={
        "max_length": 786,  # Default max length
        "min_length": 20,   # Default min length
        "num_beams": 6,
        "length_penalty": 1.0 # (balance between long and short summary )
    }
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

# Abstractive Summarization with LangChain
def abstractive_summarization_langchain(text, min_length):
    # Updating the LLM's min_length parameter dynamically (in case of shorter summaries)
    llm.model_kwargs["min_length"] = min_length
    summary = llm_chain.run({"text": text})
    return summary

# LangChain Pipeline: Combining LexRank and LangChain-based BART summarization
def summarization_pipeline(text, lexrank_sentences=10, min_length=20):
    lexrank_summary = extractive_summarization(text, num_sentences=lexrank_sentences)
    bart_summary = abstractive_summarization_langchain(lexrank_summary, min_length)
    return bart_summary

# ROUGE Score Calculation
def calculate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

# Extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return "\n".join(text)

# Streamlit App with Gradient Colors
st.markdown("""
    <style>
        .gradient-text {
            background: linear-gradient(-355deg, #65ff3b8c, #fec17bc9);
            -webkit-background-clip: border-box;
            color: #001a96;
            border-radius: 25px;
            font-size: 20px;
            font-weight: bold;
        }
        .gradient-header {
            background: linear-gradient(90deg, #11cb91, #fc53256b);
            -webkit-background-clip: content-box;
            color: #8b006a;
            font-size: 41px;
            font-weight: bold;
        }
        .gradient-subheader {
            background: linear-gradient(90deg, #00acffd1, #9575cda6);
            -webkit-background-clip: border-box;
            color: #283593f0;
            font-size: 70px;
            border-radius: 25px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Custom Gradient for Main Title
st.markdown("<h1 class='gradient-header' style='text-align: center;'>👉Text Summarization Tool👈</br>😊📚</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='gradient-subheader' style='text-align: center;'>🤖 LangChain 🐦</h2>", unsafe_allow_html=True)
st.markdown("<h4 class='gradient-text' style='text-align: center;'> Modular pipeline (extractive + abstractive) text summarization and optional ROUGE score evaluation.📃</h4>", unsafe_allow_html=True)

# Step 1: Ask user for input method
input_method = st.radio("Do you have Text or a DOCX file?", options=["Text 🔤", "DOCX 📄"])

# Step 2: Collect user input
user_text = ""

if input_method == "Text 🔤":
    user_text = st.text_area("Enter the text to summarize:", height=200)
elif input_method == "DOCX 📄":
    uploaded_file = st.file_uploader("Upload your DOCX file:", type=["docx"])
    if uploaded_file is not None:
        with st.spinner("Extracting text from DOCX..."):
            user_text = extract_text_from_docx(uploaded_file)
            st.success("Text extracted successfully! Proceed to summarization.")

# Summarization settings
if user_text:
    has_reference = st.radio("Do you have a reference summary?", options=["No", "Yes"], index=0)
    reference_summary = None
    if has_reference == "Yes":
        reference_summary = st.text_area("Enter the reference summary:", height=100)

    # Summarization button
    if st.button("Summarize Text"):
        with st.spinner("Processing...Please wait🫷🏻⏳"):
            # Preprocess the text into manageable chunks
            preprocessed_chunks = preprocess_text(user_text)

            # Summarize each chunk and combine
            final_summaries = []
            for chunk in preprocessed_chunks:
                summary = summarization_pipeline(chunk, min_length=20 if reference_summary else 249)
                final_summaries.append(summary)

            final_summary = " ".join(final_summaries)

        # Display the generated summary
        st.subheader("Generated Summary:")
        st.write(final_summary)

        # If reference summary is provided, calculate and display ROUGE scores
        if reference_summary:
            with st.spinner("Calculating ROUGE scores..."):
                rouge_scores = calculate_rouge(reference_summary, final_summary)

            st.subheader("ROUGE Scores:")
            for metric, score in rouge_scores.items():
                st.write(f"**{metric.upper()}**: Precision: {score.precision:.4f}, Recall: {score.recall:.4f}, F1 Score: {score.fmeasure:.4f}")
        else:
            st.write("No reference summary provided. ROUGE score calculation skipped.")

# Displaying Sidebar Image and Creater Name

with st.sidebar:
    st.image("https://i.graphicmama.com/uploads/2023/3/6414793b7befa-602a4f0ed6583-Flying%20Robot%20Cartoon%20Animated%20GIFs.gif", use_container_width=True)
    st.markdown("<center>Made with ❤️ Kamran Mustafa 😊<br>© copyright 2024 - 2025</center>", unsafe_allow_html=True)
    st.markdown("<center>🎓 Birmingham City University 🎓</center>", unsafe_allow_html=True)



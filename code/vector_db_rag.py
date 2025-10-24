import logging
import os
import sys
import io
import streamlit as st
from prompt_builder import build_prompt_from_config
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from utils import load_yaml_config
from paths import APP_CONFIG_FPATH, OUTPUTS_DIR, PROMPT_CONFIG_FPATH
from vector_db_ingest import embed_documents, get_db_collection

# Initialize logging
logger = logging.getLogger()


def setup_logging():
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        os.path.join(OUTPUTS_DIR, "vector_db_rag.log"),
        encoding="utf-8"
    )
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def initialize_app():
    """Initialize the application with configurations and database."""
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    setup_logging()

    app_config = load_yaml_config(APP_CONFIG_FPATH)
    prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)
    collection = get_db_collection(collection_name="rule_books")

    return app_config, prompt_config, collection


def retrieve_relevant_documents(
        collection,
        query: str,
        n_results: int = 5,
        threshold: float = 0.3,
):
    """Retrieve relevant documents from the vector database."""
    relevant_results = {
        "ids": [],
        "documents": [],
        "metadatas": [],
        "distances": [],
    }

    logging.info("Embedding query...")
    embed_query = embed_documents([query])[0]

    logging.info("Querying collection...")
    results = collection.query(
        query_embeddings=[embed_query],
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )

    logging.info("Filtering results...")
    keep_item = [False] * len(results["ids"][0])
    for i, doc_distance in enumerate(results["distances"][0]):
        if doc_distance < threshold:
            keep_item[i] = True

    for i, keep in enumerate(keep_item):
        if keep:
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(results["documents"][0][i])
            relevant_results["metadatas"].append(results["metadatas"][0][i])
            relevant_results["distances"].append(results["distances"][0][i])

    return relevant_results["documents"], relevant_results["metadatas"]


def respond_to_query(
        collection,
        prompt_config: dict,
        query: str,
        llm: str,
        threshold: float = 0.3,
        n_results: int = 5
) -> str:
    """Generate response to user query using RAG."""
    docs, metas = retrieve_relevant_documents(
        collection, query, n_results=n_results, threshold=threshold
    )

    relevant_documents = (
        "\n\n".join(
            f"chapter {meta['chapter_number']}: {meta['chapter_title']}\n"
            f"source document: {meta['source_document']}\n"
            f"content: {doc}"
            for doc, meta in zip(docs, metas)
        )
    )

    logging.info("-" * 100)
    logging.info("Relevant documents prepared:\n%s", relevant_documents)
    logging.info("")
    logging.info("User's question:")
    logging.info(query)
    logging.info("")
    logging.info("-" * 100)
    logging.info("")

    input_data = (
        f"Relevant documents:\n\n{relevant_documents}\n\nUser's question:\n\n{query}"
    )
    prompt = build_prompt_from_config(prompt_config, input_data)

    logging.info(f"RAG assistant prompt: {prompt}")
    logging.info("")

    llm_instance = ChatGroq(model=llm)
    response = llm_instance.invoke(prompt)

    logging.info("-" * 100)
    logging.info("LLM response:")
    logging.info(response.content + "\n\n")

    return response.content


def main():
    st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")
    st.title("🤖 RAG Assistant")

    # Initialize session state
    if 'initialized' not in st.session_state:
        with st.spinner("Initializing application..."):
            app_config, prompt_config, collection = initialize_app()
            st.session_state.app_config = app_config
            st.session_state.prompt_config = prompt_config
            st.session_state.collection = collection
            st.session_state.initialized = True
            st.session_state.response = ""

            # Set default vectordb params
            vectordb_params = app_config.get("vectordb", {})
            st.session_state.threshold = vectordb_params.get("threshold", 0.3)
            st.session_state.n_results = vectordb_params.get("n_results", 5)

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        threshold = st.number_input(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.threshold,
            step=0.05,
            help="Distance threshold for filtering results"
        )

        n_results = st.number_input(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=st.session_state.n_results,
            step=1,
            help="Number of documents to retrieve"
        )

        if st.button("💾 Save Configuration", use_container_width=True):
            st.session_state.threshold = threshold
            st.session_state.n_results = n_results
            st.success("Configuration saved!")

        st.divider()

        if st.button("🚪 Quit Application", use_container_width=True, type="secondary"):
            st.info("Close the browser tab to quit the application.")

    # Main content area
    st.subheader("📝 Your Question")

    # Query input with submit button side by side
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input(
            "Enter your question:",
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )

    with col2:
        submit_button = st.button("🔍 Submit", use_container_width=True, type="primary")

    if submit_button:
        if query.strip():
            with st.spinner("Generating response..."):
                try:
                    response = respond_to_query(
                        collection=st.session_state.collection,
                        prompt_config=st.session_state.prompt_config["rag_assistant_prompt"],
                        query=query,
                        llm=st.session_state.app_config["llm"],
                        threshold=st.session_state.threshold,
                        n_results=st.session_state.n_results
                    )
                    st.session_state.response = response
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    logging.error(f"Error: {str(e)}", exc_info=True)
        else:
            st.warning("Please enter a question before submitting.")

    # Response section below the query
    st.subheader("💬 LLM Response")
    response_container = st.container(height=400)
    with response_container:
        if st.session_state.response:
            st.markdown(st.session_state.response)
        else:
            st.info("Response will appear here after submitting a query.")

    # Display current configuration
    st.divider()
    st.caption(f"Current settings: Threshold = {st.session_state.threshold}, Results = {st.session_state.n_results}")


if __name__ == "__main__":
    main()
#final version
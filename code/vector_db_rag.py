import logging
import os
import sys
import io
from prompt_builder import build_prompt_from_config
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from utils import load_yaml_config
from paths import APP_CONFIG_FPATH, OUTPUTS_DIR, PROMPT_CONFIG_FPATH
from vector_db_ingest import embed_documents, get_db_collection

logger = logging.getLogger()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def setup_logging():
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(OUTPUTS_DIR, "vector_db_rag.log"), encoding="utf-8")

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

collection = get_db_collection(collection_name="rule_books")


def retrieve_relevant_documents(
        query: str,
        n_results: int = 5,
        threshold: float = 0.3,
):
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
        prompt_config: dict,
        query: str,
        llm: str,
        threshold: float = 0.3,
        n_results: int = 5

) -> str:
    docs, metas = retrieve_relevant_documents(query, n_results=n_results, threshold=threshold)
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

    llm = ChatGroq(model=llm)
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    setup_logging()
    app_config = load_yaml_config(APP_CONFIG_FPATH)
    prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)

    rag_assistant_config = prompt_config["rag_assistant_prompt"]
    llm = app_config["llm"]
    vectordb_params = app_config["vectordb"]

    exit_app = False
    while not exit_app:
        query = input(
            "Enter a question, 'config' to change the parameters, or 'exit' to quit: "
        )

        if query == "exit":
            exit_app = True
            exit()

        if query == "config":
            threshold = float(input("Enter the threshold value: "))
            n_results = int(input("Enter the number of results: "))
            vectordb_params = {
                "n_results": n_results,
                "threshold": threshold
            }
            continue

        response = respond_to_query(
            prompt_config=rag_assistant_config,
            query=query,
            llm=llm,
            **vectordb_params
        )

        logging.info("-" * 100)
        logging.info("LLM response:")
        logging.info(response + "\n\n")

# console working version
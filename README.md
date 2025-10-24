# Rules-of-the-game-RAG

As a fan of football (soccer) one of the most annoying parts of watching a game are referees who confuse everybody with their decisions.

So I created a RAG Web application, where I utilize LLMs via GrokAPI, for questions and answers on the IFAB rules.
IFAB stands for 
International Football Association Board,
and they are determining the Laws of the Game (divided into files for each chapter in the folder data).

Features of the application
1. Natural language  querying of the IFAB rules (for season 2025/26)
2. Chapter reference and resources (for transparency)
3. Changeable (configurable) retrival depth and threshold

The Retrival-Augmented-Generation (RAG) pipeline consists of:
1. ChromaDb vector database (storing laws)
2. HuggingFace MiniLM embeddings for similarity search
3. LLM answers via GrokAPI

How does it all work:
1. The rule books (chapters) are loaded and split into chunks long 1000 characters with 200 characters overlap
2. The chunks are encoded and later stored in ChromaDB vector database
3. After entering, the natural language question (query) is embedded and matched against relevant rules (with the adequate chapter,source etc.)
4. A RAG prompt is constructed and send to the LLM, which generates the final answer

How to use it locally:
1. Clone this repository
2. Create a virtual environment 
3. Install the dependencies from requirements.txt
4. Create the .env file where you will store your GROQ_API_KEY
5. run vector_db_ingest.py
6. streamlit run vector_db_rag.py (or absolute file path) in your console

What can you change when using it locally:
1. Embedding model 
2. Used LLM inside of config.yaml
3. Reasoning strategies and memory strategies
4. Configurations when running the project or prior to running in yaml.config
5. The prompt_config.yaml file contend
6. Clone the repository upto second commit for using the version where answers are displayed in the console
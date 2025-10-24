import os


ROOT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_PATH=os.path.join(ROOT_DIR,'.env')

CODE_DIR=os.path.join(ROOT_DIR,'code')

APP_CONFIG_FPATH = os.path.join(CODE_DIR, "config", "config.yaml")
PROMPT_CONFIG_FPATH = os.path.join(CODE_DIR, "config", "prompt_config.yaml")

OUTPUTS_DIR=os.path.join(ROOT_DIR,'outputs')

DATA_DIR=os.path.join(ROOT_DIR,'data')

VECTOR_DB_DIR=os.path.join(OUTPUTS_DIR,"vector_db")

METADATA_FPATH=os.path.join(DATA_DIR,"metadata.yaml")
#defining all the needed paths 
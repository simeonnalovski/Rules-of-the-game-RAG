import os
from pathlib import Path
from typing import Union
from pypdf import PdfReader
import yaml
from paths import DATA_DIR

def get_pdf_text_as_string(pdf_path: Path) -> str:
  Reader=PdfReader(pdf_path)
  text_content=""
  try:   
    for page in Reader.pages:
      text_content+=page.extract_text() or ""
  except Exception as e:
    print(f"Error reading {pdf_path}: {e}")
  return text_content


def load_rule_book(rule_book_external_id="01 - The Ball"):
  rule_book_path=Path(DATA_DIR)/f"{rule_book_external_id}.pdf"
  if not rule_book_path.exists():
    raise FileNotFoundError(f"Rule book {rule_book_external_id} not found")
  return get_pdf_text_as_string(rule_book_path)

def load_all_rule_books(rule_book_dir:str=DATA_DIR) -> list[str]:
  rule_books=[]
  for book in os.listdir(rule_book_dir):
    if book.endswith(".pdf"):
      rule_books.append(load_rule_book(Path(book).stem))
  return rule_books

def load_yaml_config(file_path: Union[str, Path]) -> dict:
    
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path}")

    # Read and parse the YAML file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e
    except IOError as e:
        raise IOError(f"Error reading YAML file: {e}") from e

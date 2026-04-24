from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

# ---------- MODEL ----------
model = ChatMistralAI(model="mistral-small-2603")

# ---------- SCHEMA ----------
class MovieInfo(BaseModel):
    Title: str
    Genre: List[str]
    Release_Year: Optional[int]
    Rating: Optional[str]
    Director: str
    Cast: List[str]
    Plot: str
    Key_Themes: List[str]
    Summary: str

parser = PydanticOutputParser(
    pydantic_object=MovieInfo,
    validate_template=False,
    output_format="json"
)

# ---------- HUMAN PROMPT ----------
human_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert movie information extractor and summarizer.

Your task is to carefully read a paragraph about a movie and extract meaningful structured information in a clean, readable format.

STRICT RULES:
- Only extract information explicitly present in the paragraph
- Do NOT hallucinate or guess missing details
- If something is not mentioned, write "Not Available"
- Keep answers concise and clean
- Summary must be 3-4 lines maximum
- Output should be human-readable (NOT JSON)

You must follow the exact format below:

Movie Name:
Genre:
Release Year:
Director:
Cast:
Plot:
Key Themes:
Rating:
Notable Features:
Short Summary:

Formatting Rules:
- Cast should be comma-separated
- Key Themes should be comma-separated
- No extra text before or after
"""),
    ("human", "Extract from:\n\n{paragraph}")
])

# ---------- JSON PROMPT ----------
json_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert movie information extractor.

STRICT RULES:
- ONLY return valid JSON
- No extra text
- Follow schema exactly
- Missing → "NA"

{format_instructions}
"""),
    ("human", "Extract from:\n\n{paragraph}")
])

# ---------- INPUT ----------
mode = input("Select mode (1 = Human, 2 = JSON): ")
para = input("Enter movie description: ")

# ---------- LOGIC ----------
if mode == "1":
    # HUMAN MODE
    final_prompt = human_prompt.invoke({
        "paragraph": para
    })

    response = model.invoke(final_prompt)
    print("\n🎬 Human Readable Output:\n")
    print(response.content)

else:
    # JSON MODE
    final_prompt = json_prompt.invoke({
        "paragraph": para,
        "format_instructions": parser.get_format_instructions()
    })

    response = model.invoke(final_prompt)

    try:
        data = parser.parse(response.content)

        print("\n🤖 Structured JSON Output:\n")
        print(data.json(indent=2))

    except Exception as e:
        print("\n⚠️ Parsing failed. Raw output:\n")
        print(response.content)
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatMistralAI(model = "mistral-small-2603")
prompt = ChatPromptTemplate.from_messages(
    [    ("system", """
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

("human", """
Extract the information from the paragraph below:

{paragraph}
""")
])

para = input("Enter a movie description paragraph: ")

final_prompt = prompt.invoke({
    "paragraph" : para
})

response = model.invoke(final_prompt)
print(response.content)
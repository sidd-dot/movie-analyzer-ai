from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

model = ChatMistralAI(model = "mistral-small-2603")

class MovieInfo(BaseModel):
    Title: str
    Genre: List[str]
    Release_Year: Optional[int]
    Rating: Optional[str]
    Director:str
    Cast: List[str]
    Plot: str
    Key_Themes: List[str]
    Summary: str


parser = PydanticOutputParser(pydantic_object=MovieInfo, validate_template=False, output_format="json")



prompt = ChatPromptTemplate.from_messages(
    [    ("system", """
You are an expert movie information extractor and summarizer.
          
{format_instructions}
"""),

("human", """
Extract the information from the paragraph below:

{paragraph}
""")
])

para = input("Enter a movie description paragraph: ")

final_prompt = prompt.invoke({
    "paragraph" : para,
    "format_instructions" : parser.get_format_instructions()
})

response = model.invoke(final_prompt)
movie_data = parser.parse(response.content)
print(movie_data)
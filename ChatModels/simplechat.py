from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
model =  ChatMistralAI(model = "mistral-small-2603")

message = [

]

print ("----------------------------- Welcome to the Chat Models -----------------------------")                 
while True:
    prompt = input("You: ")
    message.append(prompt)
    response = model.invoke(prompt)
    if prompt == "0":
        break
    message.append(response.content)
    print("Bot :", response.content)


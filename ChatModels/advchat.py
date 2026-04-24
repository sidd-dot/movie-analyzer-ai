from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

message = [
    SystemMessage(content = "YOU ARE A HELPFUL ASSISTANT.")
]

from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model = "mistral-small-2603")

print ("----------------------------- Welcome to the Chat Models -----------------------------")

while True:
    prompt = input("You: ")
    message.append(HumanMessage(content = prompt))
    response = model.invoke(message)
    if prompt == "0":
        break
    message.append(AIMessage(content = response.content))
    print("Bot :", response.content)

print(message)

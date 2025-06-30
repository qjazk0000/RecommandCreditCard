from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()
result = llm.invoke("Hello, world!")

print(result.content)
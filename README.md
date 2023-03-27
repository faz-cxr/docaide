# 🦜️🔗 ChatLangChain

This repo is an implementation of a locally hosted chatbot specifically focused on question answering over the [CKS NICE Guidelines](https://cks.nice.org.uk).
Built with [LangChain](https://github.com/hwchase17/langchain/) and [FastAPI](https://fastapi.tiangolo.com/).

The app leverages LangChain's streaming support and async API to update the page in real time for multiple users.

## 💻 Demo

![Demo Screen recording](assets/images/demo.gif)

## ✅ Running locally
- Install dependencies: `pip install -r requirements.txt`
- Run the app: `make start`
   - To enable tracing, make sure `langchain-server` is running locally and pass `tracing=True` to `get_chain` in `main.py`. You can find more documentation [here](https://langchain.readthedocs.io/en/latest/tracing.html).
- Open [localhost:9000](http://localhost:9000) in your browser.

## 🚀 Important Links

Deployed version (to be updated soon): [Coming soon](https://example.com)

## 📚 Technical description

Question-Answering has the following steps, all handled by [ConversationalRetrievalChain](https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html):

1. Given the chat history and new user input, determine what a standalone question would be (using GPT-3.5-turbo).
2. Given that standalone question, look up relevant documents from the vectorstore.
3. Pass the standalone question and relevant documents to GPT-3.5-turbo to generate a final answer.
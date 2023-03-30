"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.vectorstores.base import VectorStore

import promptlayer
promptlayer.api_key = "pl_2b1769e7202b6a141d4491fca41e308a"


def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ConversationalRetrievalChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = PromptLayerChatOpenAI(
        pl_tags=["chat-nice-ws-qgen"],
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = PromptLayerChatOpenAI(
        pl_tags=["chat-nice-ws-stream"],
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )
    template = """You are given the following extracted parts of a long document and a question, provide a conversational answer.
If the question includes a request for code, provide a code block directly from the documentation.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question can't be answered from the extracted parts, politely inform them that you are tuned to only answer questions about NICE guidelines fed to you.

ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be hyperlinks that are explicitly listed as a source in the context. Do NOT make up a hyperlink that is not listed.
Example of your response should be:

```
The answer is foo

SOURCES: https://cks.nice.org.uk/xyz

```

# Extracted documents:
---
{summaries}
---

Question: {question}"""
    QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "summaries"])
    _template = """Given the following conversation and a follow up question. Frame a standalone question combining all information from the Chat History with the Follow Up Input.
Please ignore the SOURCES.
Chat History:
---
{chat_history}
---
Follow Up Input: {question}
Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )

    doc_chain = load_qa_with_sources_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    )

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
    )
    return qa
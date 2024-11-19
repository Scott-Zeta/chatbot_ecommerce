from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_openai import ChatOpenAI
from .ingest import ingestdata


def generation(vstore):
    def retriever(question):
        results = vstore.similarity_search(question)
        # print(f"Question: {question}")
        # print(f"Results: {results}")
        if len(results) > 0:
            context = []
            for res in results:
                context.append(f"{res.metadata['content']}")
            # print(f"Context: {context}")
            return context
        return ""
    
    PRODUCT_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses, and provide url to the product.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    
    """


    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)

    llm = ChatOpenAI()

    chain = (
        RunnableMap({
            "context": retriever,
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if __name__=='__main__':
    vstore = ingestdata("done")
    chain  = generation(vstore)
    print(chain.invoke("Is there any products good for halloween?"))
    
    
    
    
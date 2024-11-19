import pandas as pd
from langchain_core.documents import Document



def dataconveter():
    # import os
    # print("Current working directory:", os.getcwd())
    product_data=pd.read_csv("./data/products.csv")

    data=product_data.copy()

    docs = []
    for index, row in data.iterrows():
        page_content = f"Product: {row['Title']} | Description: {row['Description']} | Tags: {row['Tags']} | Price: {row['Price']}"
        metadata = {"content": f"Product: {row['Title']} | Description: {row['Description']} | URL: {row['URL']} | Price: {row['Price']}"}
        doc = Document(page_content=page_content, metadata=metadata)
        docs.append(doc)
    return docs

if __name__ == "__main__":
    print(f"Meatdata: {dataconveter()[0].metadata}")
    print(f"Page_content: {dataconveter()[0].page_content}")
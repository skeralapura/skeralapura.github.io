---
layout: post
title: Building an Retail Store Assistant Using RAG (Retrieval Augmented Generation) and MCP
image: "/img/posts/gen-ai-rag-title-img.png"
tags: [GenAI, RAG, LLMs, Python, LangChain, MCP]
---

This project showcases a simple Retail Store Agent (Assistant) capable of answering customer questions catering to Store FAQs and questions related to customer's orders using **Retrieval Augmented Generation (RAG)** and **MCP**.

A core RAG system that loads internal documents, chunks them intelligently, embeds them into a vector database, retrieves relevant content, and returns it to the agent.  

A MCP client connection to a PostgreSQL database hosted on AWS which stores data related to Store's orders including products sold, sales amount and shipping status.

A agent is setup to take the customer query, evaluate and route to the correct tool or MCP and generate grounded answers.

**Key Objectives/Goals:**
- Understand the building blocks for a multi-tool agentic system which can call inbuilt tools or external MCP servers.
- Understand how a end to end RAG system works
- Considerations for prompt building
- Understand how to instantiate and call MCP servers

# Table of Contents

- [01. System Design](#system-design)
- [02. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [03. Data Overview](#data-overview)
- [04. Building the Core RAG System](#rag-core)
    - [Secure API Handling](#rag-api)
    - [Document Loading](#rag-docs)
    - [Document Chunking](#rag-chunking)
    - [Embeddings & Vector Store](#rag-embeddings)
    - [LLM Setup](#rag-llm)
    - [Prompt Template](#rag-prompt)
    - [Retriever Setup](#rag-retriever)
    - [Full RAG Pipeline](#rag-pipeline)
- [05. SQL MCP Tool](#sql-mcp)
- [06. Building the Agent](#agent-core)
    - [Establishing the MCP Connection](#mcp-conn)
    - [Tool Discovery & SQL Agent creation](#sql-agent)
    - [Creating the "Sales DB" Wrapper](#db-tool)
    - [The Parent (Orchestrator) Agent](#parent-agent)
    - [Execution](#agent-exec)
- [07. Growth & Next Steps](#growth-next-steps)

___

# 01. System Design <a name="system-design"></a>

![System Design](/img/posts/Retail_Store_Assistant_System_Diagram.jpg)

___

# 02. Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

A retail store operates a busy customer help-desk, answering queries around store hours, customer orders, delivery services, and general store operations.

They need an **AI assistant** that can answer these questions accurately, consistently, and safely, using only approved internal information.

### Actions <a name="overview-actions"></a>

Build a full end-to-end Agentic system which includes a RAG system that:

* Loads internal FAQ documentation  
* Split it into meaningful chunks  
* Created vector embeddings  
* Stored these embeddings in a persistent vector database  
* Retrieved only the most relevant content at query time  
* Generated answers grounded strictly in this retrieved context  

And a MCP tool that:

* Sets up a MCP client and connection
* Format a SQL query based on customer's question
* Retrieve customer order data

Finally a parent (orchestrator) agent which calls either the RAG tool or the SQL MCP tool based on user's query.
Internally, added monitoring, tracing, and evaluation using LangSmith during development.

### Results <a name="overview-results"></a>

The final assistant:

* Routes to the correct tool or MCP based on customer's question 
* Reliably answers customer help-desk questions and questions related their orders
* Grounds every answer in retrieved internal documentation  
* Rejects unsupported questions with a safe fallback message  
* Prevents hallucinations using strict grounding rules  

### Growth/Next Steps <a name="overview-growth"></a>

Potential future enhancements include:

* Ingestion of multiple document types (PDFs, product catalogues)  
* Add conversational memory and context/history tracking  
* Adding a real chat interface (frontend + backend)  
* Adding additional features like refunds, customer data lookups
* Building automated document ingestion pipelines for the RAG system 
* Agent and LLM Evaluation strategies 

___

# 01. Data Overview <a name="data-overview"></a>

The dataset for the RAG system contains **many question–answer pairs** in markdown format taken from Retail Store’s internal help-desk documentation.

Each Q&A pair follows a consistent structure, which can be seen in below examples:

```md
### 0001
Q: What is Retail Grocery?
A: Retail Grocery is a nationwide supermarket chain that sells fresh groceries and non-food items such as clothes, shoes and office supplies.

### 0012
Q: Do you offer home delivery?
A: Yes. We offer home delivery 7 days a week. Delivery fees and times depend on location.

### 0020
Q: What are your store hours? Do they change on holidays?
A: Most locations are open 7am–10pm daily. Holiday hours may differ by location—check the Store Locator for exact opening times on specific dates (e.g., Thanksgiving, Christmas, New Year’s Day).
```

___


# 02. Building the Core RAG System <a name="rag-core"></a>

<br>
## Secure API Handling <a name="rag-api"></a>

Load API keys from a **.env** file. This prevents credentials from being hard-coded directly in the script.

```python
from dotenv import load_dotenv
load_dotenv()
```

---


## Document Loading <a name="rag-docs"></a>

Use LangChain’s `TextLoader` to import our help-desk markdown file.

```python
from langchain_community.document_loaders import TextLoader

raw_filename = 'retail-store-help-desk-data.md'
loader = TextLoader(raw_filename, encoding="utf-8")
docs = loader.load()
text = docs[0].page_content
```

<br>
**Why this matters:**  Document loaders standardise the data into LangChain *Document* objects, which makes later steps like chunking and embedding seamless.

---

## Document Chunking <a name="rag-chunking"></a>

Split the markdown by level-3 headers (`###`), where each header introduces a new Q&A pair.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("###", "id")],
    strip_headers=True
)

chunked_docs = splitter.split_text(text)
print(len(chunked_docs), "Q/A chunks")
```

<br>
**Why this matters:**  Chunking ensures retrieval focuses on the specific Q&A pair that relates to a user query.  Good chunking dramatically improves retrieval accuracy.

---

## Embeddings & Vector Store <a name="rag-embeddings"></a>

Embeddings convert text into **numeric vectors** that represent meaning.  Documents with similar meaning end up closer together in vector space.

Embed each Q&A chunk and store the embeddings in Chroma:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="retail_vector_db_chroma",
    collection_name="retail_help_qa")
```

<br>
To load later, instead of re-creating from scratch, we can use this code:

```python
vectorstore = Chroma(
    persist_directory="retail_vector_db_chroma",
    collection_name="retail_help_qa",
    embedding_function=embeddings)
```

---

## LLM Setup <a name="rag-llm"></a>

Instantiate the model that will generate the final answer:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1", temperature=0)
```

<br>
A temperature of 0 is essential for help-desk systems where consistency and accuracy matter more than creativity.

---

## Prompt Template <a name="rag-prompt"></a>

The prompt instructs the model to answer **only** using retrieved context, and to avoid hallucination. Sample below:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
"""
System Instructions: You are a helpful assistant for Retail Store - your job is to find the best solutions & answers for the customer's query.
Answer ONLY using the provided context. If the answer is not in the context, say that you don't have this information and encourage the customer to email human@retail-store.com

Context: {context}

Question: {question}

Answer:
"""
)
```

<br>
**Why this matters:**  Prompt templates are the *instructions* that govern how the LLM behaves.  They ensure the assistant is safe, grounded, and consistent.

We have kept this simple here, but have included one important instruction for the LLM: that if the answer is not in the context, to say that it doesn't have this information and to encourage the customer to email human@retail-store.com

---

## Retriever Setup <a name="rag-retriever"></a>

Configure how relevant chunks are selected from the vector database:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 6, "score_threshold": 0.25})
```

<br>
This retrieval is set up in a way where it will retrieve *up to* 6 documents, but only if they meet the specified relevance score threshold of 0.25. 
<br>
This keeps the context focused and prevents irrelevant content from confusing the LLM.

---

## Full RAG Pipeline <a name="rag-pipeline"></a>

This pipeline connects all of the key components of the system, namely:

1. Take in the user query  
2. Retrieve in relevant chunks from the vector database  
3. Format them 
4. Inject them into the prompt template, along with the system instructions and user query 
5. Pass this information to the LLM  
6. Return the answer  

RAG answer chain: {input} -> retrieve -> format -> prompt -> model -> string
In the context of a RAG pipeline, RunnableLambda(format_docs) is a bridge that transforms the raw output of a retriever (list of document objects) into a format the LLM can understand. RunnableLambda does Type Conversion + Chain Integration + allows Observability in langsmith - this formatting step will show up as its own named block, allowing you to see exactly what "context" was sent to the prompt after retrieval
Place this inside a function and annotate it as a tool for Agent to use. Langchain "Tool" is used for this.

```python
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from langchain_core.tools import tool

def format_docs(docs):
    contents = [d.page_content for d in docs]
    return "\n\n".join(contents)

@tool
def rag_agent(question: str):
    """Use this tool for questions about store policies, returns, or general FAQs."""
    
    rag_answer_chain = (
        {
            "context": itemgetter("input") | retriever | RunnableLambda(format_docs),
            "question": itemgetter("input"),    
        }
        | prompt
        | llm
    )

    response = rag_answer_chain.invoke({"input": question})
    return response.content
```
<br>

___

# 03. SQL MCP Tool <a name="sql-mcp"></a>

Initialize parameters for calling MCP server.

```python
from mcp import ClientSession, StdioServerParameters

# Fetch MCP parameters
fs_params = StdioServerParameters(
    command="npx", 
    args=["-y", "@modelcontextprotocol/server-postgres", f"postgresql://postgres:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DBNAME')}?sslmode=no-verify"]
)
```

---

## SQL Prompt Template <a name="sql-prompt"></a>

Bring in the system prompt for SQL agent. This template contains instructions for the SQL Agent to answer **only** using the specified role and scope. The template also contains metadata information regarding the database tables and columns, relationships between tables as well as sample SQL queries.

```python
with open("sql-system-prompt.txt", "r", encoding="utf-8") as f:
    sql_system_text = f.read()
```

___

# 04. Building the Agent <a name="agent-core"></a>

Now we build the main engine for the parent (core) agent. 

## 4a. Establishing the MCP Connection <a name="mcp-conn"></a>
The first two lines use stdio_client and ClientSession to create a communication bridge. This allows the Python script to talk to an MCP Server (like a database connector).


```python
from mcp.client.stdio import stdio_client

async with stdio_client(fs_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
```

---

## 4b. Tool Discovery & SQL Agent creation <a name="sql-agent"></a>

- load_mcp_tools: Automatically fetches the available database functions (like "read schema" or "run query") from the MCP server.
- sql_agent: A specialized "worker" agent is created. Its only job is to handle SQL-related tasks using those MCP tools.

```python
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

        mcp_tools = await load_mcp_tools(session)

        sql_agent = create_agent(
            llm, 
            tools=mcp_tools, 
            system_prompt=sql_system_text
        )
```

---

## 4c. Creating the "Sales DB" Wrapper <a name="db-tool"></a>

The @tool decorator transforms the sql_agent into a single tool called sales_db_tool. When this tool is called, it passes the user's question to the SQL agent. It abstracts away the complexity—the parent agent doesn't need to know how to write SQL; it just asks the "Sales DB" tool for help

```python

        @tool
        async def sales_db_tool(query: str):
            """Use this for order status, shipping status or sales data."""

            try:

                response = await sql_agent.ainvoke({"messages": [("user", query)]})
                return response["messages"][-1].content
            except Exception as e:
                return f"Error accessing database: {str(e)}"
```

---

## 4d. The Parent (Orchestrator) Agent <a name="parent-agent"></a>

The parent_agent acts as a router. It has two tools:
- sales_db_tool: For orders and shipping (the SQL worker)
- rag_agent: For store policies or FAQs (a separate knowledge worker)

```python
        # Initialize the parent Agent
        parent_agent = create_agent(llm, 
                                    tools=[sales_db_tool, rag_agent],
                                    system_prompt="You are a retail assistant. Route queries to the order expert or policy expert as needed.")
```

---

## 4e. Execution <a name="agent-exec"></a>
Call the parent agent and pass the user query. The agent interprets the user query and calls the appropriate tool and returns answers back to user.

```python
        # Example 1: Routes to PostgreSQL MCP
        res1 = await parent_agent.ainvoke({"messages": [("user", "What is the shipping status for Order_ID = 'CA-2019-103800'?")]})
        print(f"Order Query: {res1['messages'][-1].content}")

        # Example 2: Routes to RAG Tool
        res2 = await parent_agent.ainvoke({"messages": [("user", "How do I return an item?")]})
        print(f"Customer Support Query: {res2['messages'][-1].content}")
```

___

# 05. Growth & Next Steps <a name="growth-next-steps"></a>

Potential future enhancements include:

* Ingestion of multiple data types (PDFs, product catalogues)  
* Integrating SQL tools for real-time store data, delivery slots, or loyalty information  
* Building a production web interface (React + FastAPI)  
* Automated indexing pipelines to detect new documents  
* Response streaming for real-time chat UX  

___

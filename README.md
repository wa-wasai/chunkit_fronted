# Readme

Our team’s RAG library （Since I’m trying to get better at English writing, please allow me to write in English.)

---

## **Function**:



### Initialize the knowledge base：

1. Put the folder you want to initialize into vector database at the same level as “builder.py”
2. Run 'builder.py' and then input the folder name
3. Only support .docx, .pdf and .txt

### RAG api

1. Import *RAG* from **RAGlibrary**

```python
from RAGlibrary import RAG
```

2. call example

```python
from RAGlibrary import RAG
query = input()
rag = RAG()
output = rag.call_RAG(query) #Non-streaming output

for delta in rag.call_RAG_stream(query): #streaming output
    print(f'{delta}')
```

---

## **Installation and Execution**



### Setting Environment Variables

1. We use the LLM service from Bailian. To get started, create an empty **Application** in Bailian. You only need to select the model without configuring other settings—especially ensure the system prompt remains empty. Once done, set your API key and app ID as environment variables by running this code in PowerShell:

```powershell
(System.Environment)::SetEnvironmentVariable("DASHSCOPE_API_KEY", "##your api key", "User")
(System.Environment)::SetEnvironmentVariable("LLM_appid", "##your appid", "User")
```

### Install python dependencies

```powershell
pip install -r requirements.txt
```



### External Library Preparation

1. Install Qwen3-embedding-0.6B and put it at the same level as *RAGlibrary.py*，[点此下载](https://modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B)


### <font color="grenn">启动FastAPI服务器</font>
```bash
python .\fastapi_server_stream_only.py
```

### <font color="red">使用Python内置服务器</font>
```bash
python -m http.server 8080
```


import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def initialize_sales_bot(vector_store_dir: str = "real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    global SALES_BOT
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                            retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                      search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT


def shipping_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    if len(ans["source_documents"]) == 0:
            template = """
            以下是之前的對話：
            {history}
            客戶的最新回答是：{question}
            请重新给一个更自然，连贯的回复，要像一个真人航运顾问一样
            """
            llm = ChatOpenAI(model_name="gpt-4", temperature=0)

            prompt = PromptTemplate(template=template, input_variables=["history", "question"])
            chain = LLMChain(llm=llm, prompt=prompt)

            response = chain.run(history=formatted_history, question=message)

            return response 
    else:
        return "稍等一下，这个问题我得去找领导确认确认"


def launch_gradio():
    demo = gr.ChatInterface(
        fn=shipping_chat,
        title="航运集装箱销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    # 初始化航运集装箱机器人
    initialize_sales_bot(vector_store_dir="shipping_qa_pair")
    # 启动 Gradio 服务
    launch_gradio()

import os
import bs4
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader  # Import PDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings


class RAG:
    def __init__(self) -> None:
        self.hf = HuggingFacePipeline.from_model_id(
            model_id="gpt2",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 50},
        )

        self.hf_embeddings = HuggingFaceEmbeddings()

        self.pdf_loc_base = "./pdf/"

    def _format_docs(self, docs) -> str:
        return "".join(doc.page_content for doc in docs)

    def generate(self, file_name: str, question: str) -> str:
        pdf_path = self.pdf_loc_base + file_name
        loader = PyPDFLoader(pdf_path)  # Load PDF using PDFLoader
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=self.hf_embeddings
        )

        retriever = vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.hf
            | StrOutputParser()
        )

        string = rag_chain.invoke(question)

        s = string.split("Answer")
        return s[1][2:]

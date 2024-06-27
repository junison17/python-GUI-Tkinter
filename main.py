import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# langchain 패키지
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# RAG Chain 구현을 위한 패키지
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PyPDF2 import PdfReader

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Chain Implementation")
        self.root.geometry("800x600")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.main_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.content_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0,0), window=self.content_frame, anchor="nw")

        self.style = ttk.Style()
        self.style.configure("TLabel", font=("Helvetica", 12))
        self.style.configure("TButton", font=("Helvetica", 12))
        self.style.configure("TEntry", font=("Helvetica", 12))
        self.style.configure("TCombobox", font=("Helvetica", 12))
        self.style.configure("TScale", font=("Helvetica", 12))

        self.create_widgets()
        self.pdf_file_path = 'C:\\ragpdf\\ragpdf\\hospital.pdf'
        self.vectorstore = self.load_pdf_to_vector_store(self.pdf_file_path)

    def create_widgets(self):
        self.chunk_size_label = ttk.Label(self.content_frame, text="Chunk Size")
        self.chunk_size_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

        self.chunk_size = tk.IntVar(value=1000)
        self.chunk_size_scale = ttk.Scale(self.content_frame, variable=self.chunk_size, from_=0, to=5000, orient=tk.HORIZONTAL, length=200)
        self.chunk_size_scale.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        self.chunk_size_value = ttk.Label(self.content_frame, textvariable=self.chunk_size)
        self.chunk_size_value.grid(row=0, column=2, padx=10, pady=10, sticky=tk.W)

        self.chunk_overlap_label = ttk.Label(self.content_frame, text="Chunk Overlap")
        self.chunk_overlap_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)

        self.chunk_overlap = tk.IntVar(value=200)
        self.chunk_overlap_scale = ttk.Scale(self.content_frame, variable=self.chunk_overlap, from_=0, to=1000, orient=tk.HORIZONTAL, length=200)
        self.chunk_overlap_scale.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)
        self.chunk_overlap_value = ttk.Label(self.content_frame, textvariable=self.chunk_overlap)
        self.chunk_overlap_value.grid(row=1, column=2, padx=10, pady=10, sticky=tk.W)

        self.similarity_metric_label = ttk.Label(self.content_frame, text="Similarity Metric")
        self.similarity_metric_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)

        self.similarity_metric = tk.StringVar(value="cosine")
        self.similarity_metric_dropdown = ttk.Combobox(self.content_frame, textvariable=self.similarity_metric)
        self.similarity_metric_dropdown['values'] = ("cosine", "l2")
        self.similarity_metric_dropdown.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W)

        self.temperature_label = ttk.Label(self.content_frame, text="Temperature")
        self.temperature_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)

        self.temperature = tk.DoubleVar(value=0.0)
        self.temperature_scale = ttk.Scale(self.content_frame, variable=self.temperature, from_=0.0, to=2.0, orient=tk.HORIZONTAL, length=200)
        self.temperature_scale.grid(row=3, column=1, padx=10, pady=10, sticky=tk.W)
        self.temperature_value = ttk.Label(self.content_frame, textvariable=self.temperature)
        self.temperature_value.grid(row=3, column=2, padx=10, pady=10, sticky=tk.W)

        self.chat_label = ttk.Label(self.content_frame, text="Chat")
        self.chat_label.grid(row=4, column=0, padx=10, pady=10, sticky=tk.W)

        self.chat_text = scrolledtext.ScrolledText(self.content_frame, height=10, width=70, font=("Helvetica", 12))
        self.chat_text.grid(row=4, column=1, columnspan=2, padx=10, pady=10, sticky=tk.W)

        self.message_label = ttk.Label(self.content_frame, text="Enter your message")
        self.message_label.grid(row=5, column=0, padx=10, pady=10, sticky=tk.W)

        self.message_text = tk.Text(self.content_frame, height=5, width=70, font=("Helvetica", 12))
        self.message_text.grid(row=5, column=1, columnspan=2, padx=10, pady=10, sticky=tk.W)

        self.submit_button = ttk.Button(self.content_frame, text="Submit", command=self.process_pdf_and_answer)
        self.submit_button.grid(row=6, column=0, columnspan=3, pady=20)

        self.progress = ttk.Progressbar(self.content_frame, orient="horizontal", mode="indeterminate")
        self.progress.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky=tk.W+tk.E)

        self.progress_label = ttk.Label(self.content_frame, text="")
        self.progress_label.grid(row=8, column=0, columnspan=3, padx=10, pady=10, sticky=tk.W+tk.E)

    def load_pdf_to_vector_store(self, pdf_file, chunk_size=1000, chunk_overlap=100, similarity_metric='cosine'):
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, 
                                            embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
                                            collection_metadata = {'hnsw:space': similarity_metric})
        return vectorstore

    def retrieve_and_generate_answers(self, vectorstore, message, temperature=0):
        retriever = vectorstore.as_retriever()
        template = '''Answer the question based only on the following context:
        <context>
        {context}
        </context>
        Question: {input}
        '''
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=temperature, api_key=OPENAI_API_KEY)
        document_chain = create_stuff_documents_chain(model, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        response = rag_chain.invoke({'input': message})
        return response['answer']

    def process_pdf_and_answer(self):
        self.progress.start()
        self.progress_label.config(text="Processing, please wait...")
        self.root.update_idletasks()

        chunk_size = self.chunk_size.get()
        chunk_overlap = self.chunk_overlap.get()
        similarity_metric = self.similarity_metric.get()
        temperature = self.temperature.get()
        message = self.message_text.get("1.0", tk.END).strip()

        if not message:
            self.progress.stop()
            self.progress_label.config(text="")
            messagebox.showerror("Error", "Please provide a message.")
            return

        answer = self.retrieve_and_generate_answers(self.vectorstore, message, temperature)

        self.chat_text.insert(tk.END, f"User: {message}\n")
        self.chat_text.insert(tk.END, f"Bot: {answer}\n")
        self.chat_text.yview(tk.END)

        self.message_text.delete(1.0, tk.END)
        self.progress.stop()
        self.progress_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()

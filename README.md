# Natural Language Query Agent

The main goal of this project is to create a Natural Language Query Agent using Large Language Models (LLMs) and open-source vector indexing and storage frameworks. This agent will provide concise answers to straightforward questions within a large dataset of lecture notes and a table of LLM architectures. While the primary task is to generate clear responses based on reference texts, the project will also explore extending the agent's capabilities to handle more complex queries, including follow-up questions, conversational memory, reference citation, and processing large sets of lecture notes across multiple subjects.

The data sources utilized for this project encompass the following:

- [Stanford LLMs Lecture Notes](https://stanford-cs324.github.io/winter2022/lectures/)
- [Awesome LLM Milestone Papers](https://github.com/Hannibal046/Awesome-LLM#milestone-papers)

**Important Note**

You must request access to Llama 2 models via the Meta website and agree to share your Hugging Face account details with Meta. Ensure that your Hugging Face account email matches the one you submitted on the Meta website, as mismatched emails can result in request denial. The approval process usually takes a few minutes to a few hours.

**Requirements** 

The following tools are necessary for its proper functionality.

- [transformers](https://pypi.org/project/transformers/)
- [accelerate](https://pypi.org/project/accelerate/)
- [einops](https://pypi.org/project/einops/)
- [langchain](https://pypi.org/project/langchain/)
- [xformers](https://pypi.org/project/xformers/)
- [bitsandbytes](https://pypi.org/project/bitsandbytes/)
- [faiss-gpu](https://pypi.org/project/faiss-gpu/)
- [sentence_transformers](https://pypi.org/project/sentence-transformers/)
- [pypdf](https://pypi.org/project/pypdf/)

**[LLaMA 2](https://ai.meta.com/llama/)** 

LLaMA 2 is an advanced open-source language model, pretrained and fine-tuned on a vast dataset of 2 trillion tokens. It comes in three sizes: 7B, 13B, and 70B, each offering improvements over the previous LLaMA 1 models. Key enhancements include training on 40% more tokens and a context length of 4,000 tokens. LLaMA 2 outperforms other open-source language models in various benchmarks, including reasoning, coding, proficiency, and knowledge assessment tasks.

**[LangChain](https://python.langchain.com/docs/get_started/introduction)** 

LangChain is a powerful open-source framework designed for building applications powered by large language models. The core idea behind this library is the ability to connect various components to enable advanced functionalities centered around LLMs. LangChain consists of multiple components spread across different modules.


**[FAISS](https://github.com/facebookresearch/faiss)**

FAISS (Facebook AI Similarity Search) is a library that allows developers to quickly find similar embeddings in multimedia documents. It surpasses traditional hash-based query search engines by offering scalable similarity search capabilities. FAISS enables efficient searches of multimedia documents, even those impractical for standard SQL databases. It includes nearest-neighbor search solutions for datasets ranging from millions to billions in scale, balancing memory, speed, and accuracy. FAISS is designed to perform well in a variety of operational scenarios.


### This project is divided into three essential parts.
- **Preparing the dataset**
- **Initializing the HuggingFace text-generation pipeline**
- **Initializing the conversational chain**

### Preparing the Dataset

A significant challenge in this project is preparing the dataset. Manually retrieving the text is time-consuming and not scalable. Writing an automated script to read HTML documents may not cover all cases. Fortunately, LangChain has document readers that can directly retrieve and process text from web links and PDF links. We will use LangChain's document loaders, including:


- [WebBaseLoader](https://python.langchain.com/docs/integrations/document_loaders/web_base): Extracts text content from HTML web pages and transforms it into a document format suitable for various downstream tasks.

- [PyPDF](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf): Extracts text from PDF files

We will retrieve links to webpages and PDF files from the following .txt documents:

	ema_dataset_web_links.txt
	ema_dataset_pdf_links.txt

At the end of the process, we'll save the custom dataset as a .pkl file, an essential part of our pipeline. Refer to the ema_dataset.ipynb notebook in the ema-dataset folder for the exact implementation.

### Initializing the HuggingFace text-generation Pipeline

To set up a text-generation pipeline using Hugging Face transformers, you must initiate three vital elements:

- **A Language Model (LLM)**: We will use `meta-llama/Llama-2-7b-chat-hf`
- **A Tokenizer**: The pipeline necessitates a tokenizer responsible for translating human-readable text into token IDs readable by the LLM. The Llama 2 7B models were trained using the Llama 2 7B tokenizer
- **A stopping criteria object**: Establish stopping criteria for the model to determine when it should stop generating text. Without clear criteria, the model might continue producing text that deviates from the initial question.

  
### Initializing the Conversational Chain

Next, we initialize the `ConversationalRetrievalChain`. This chain allows you to create a chatbot with a memory feature, using a vector store to retrieve relevant information from your documents.

Additionally, to retrieve the source documents used to answer a question, set the `return_source_documents` parameter to `True` during the chain's construction:

	chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
	
You may refer to to the `ema-QA-model.ipynb` notebook present in the `Source Code` folder.

### Final Results

The final model generates articulate responses from reference texts and handles complex queries, including follow-up questions, conversational memory, reference citations, and processing extensive sets of lecture notes across various subjects.

- #### **Question-Answering and Reference Citation**

The conversation below was generated with the final model, demonstrating its ability to cite references to show that the model is not hallucinating.

	
> Enter your Query: What is a language model?
> 
> 
> A language model is a probability distribution over sequences of tokens. 
> 
> https://stanford-cs324.github.io/winter2022/lectures/introduction/
> 
> 
> Enter your Query: What is adaptability?
> 
> 
> The term "adaptable" refers to the ability of a language model to 
> adjust its performance to better suit a particular context or task. This 
> can involve modifying the model's parameters or fine-tuning the model on a 
> small set of in-context examples. 
> 
> https://stanford-cs324.github.io/winter2022/lectures/adaptation/
> 
> 
> Enter your Query: What are some milestone model architectures and papers in the last few years?
> 
> 
> Yes, here are some recent milestone models and papers in natural language 
> processing: * GPT-3 (2020): A large-scale transformer-based language model developed by OpenAI 
> that achieved state-of-the-art results on a wide range of natural language processing 
> tasks. * BERT (2018): A pre-trained language model developed by Google that 
> achieved state-of-the-art results on a wide range of natural language processing tasks, 
> including question answering, sentiment analysis, and text classification. * RoBERTa (2019): A 
> variant of BERT that was specifically designed for text classification tasks and 
> achieved state-of-the-art results on a number of benchmark datasets. * Megatron-LM (2019): 
> A large-scale transformer-based language model developed by Facebook that achieved state-of-the-art results 
> on a wide range of natural language processing tasks, including text classification, 
> sentiment analysis, and question answering. * T5 (2020): A text-to-text transformer model 
> developed by Google that achieved state-of-the-art results on a wide range of 
> natural language processing tasks, including text generation, question answering, and dialogue systems. 
> 
> 
> https://stanford-cs324.github.io/winter2022/lectures/introduction/
> 
> 
> Enter your Query: What are the layers in a transformer block?
> 
> 
> In the Transformer architecture described in the paper, there are 6 identical 
> layers in the encoder and decoder. Each layer consists of two sub-layers: 
> a multi-head self-attention mechanism and a simple, position-wise fully connected feed-forward network. 
> The output of each sub-layer is then passed through layer normalization and 
> a residual connection before being fed into the next layer. 
> 
> 
> 
> Enter your Query: Tell me about datasets used to train LLMs and how they’re cleaned
> 
> 
> The most common datasets used to train language models are those collected 
> from the internet, such as Common Crawl, and OpenWebText. These datasets are 
> curated and filtered to remove unwanted content, such as adult or offensive 
> material. Additionally, some datasets may undergo tokenization, part-of-speech tagging, and other forms 
> of preprocessing before being used for training. However, I cannot provide more 
> detailed information on the specific datasets used or their cleaning processes without 
> further context. 
> 
> https://stanford-cs324.github.io/winter2022/lectures/data/
> 
> 
> Enter your Query: thanks

- #### **Conversational Memory**

Furthermore, the model can generate responses by recalling its prior interactions, as illustrated in the following example:

> Enter your Query: What are large language models?
> 
> 
> Large language models are neural network models that are trained on large 
> amounts of text data to generate language outputs that are coherent and 
> natural-sounding. These models have become increasingly popular in recent years due to 
> their ability to perform a wide range of tasks, such as language 
> translation, text summarization, and text generation. They are often used in applications 
> such as chatbots, language translation software, and content creation tools. However, these 
> models also pose significant risks, such as the potential for misuse or 
> unintended consequences, and there are ongoing efforts to develop more responsible and 
> ethical ways of using them. 
> 
> https://stanford-cs324.github.io/winter2022/lectures/introduction/
> 
> https://stanford-cs324.github.io/winter2022/lectures/data/
> 
> 
> Enter your Query: Can you please elaborate?
> 
> 
> Of course! Large language models are trained on vast amounts of text 
> data, which allows them to learn patterns and relationships within language. They 
> use various techniques such as word embeddings, attention mechanisms, and neural networks 
> to capture the meaning and context of words and phrases. Through this 
> process, they can generate text, complete sentences, and even engage in conversation. 
> However, it's important to note that these models are not perfect and 
> can sometimes produce inappropriate or offensive content, so it's crucial to use 
> them responsibly and ethically. 
> 
> https://stanford-cs324.github.io/winter2022/lectures/introduction/
> 
> https://stanford-cs324.github.io/winter2022/lectures/data/
> 
> 
> Enter your Query: thanks

- #### **Summarization of Conversation Session**

> The model is capable of utlizing its conversational memory and capable of summarizing the conversation session as demonstrated below:
> 
> Enter your Query: What are transformers?
> 
> 
> Transformers are a type of artificial neural network that are particularly well-suited 
> for natural language processing tasks, such as language translation, text summarization, and 
> language modeling. They were introduced in a 2017 paper by Vaswani et 
> al. and have since become a widely-used and influential architecture in the 
> field of AI. 
> 
> https://stanford-cs324.github.io/winter2022/lectures/modeling/
> 
> 
> Enter your Query: What is prompting?
> 
> 
> In the context of transformers, prompting refers to the process of constructing 
> a prompt (a string based on the description of a task) that 
> can be used to help the model generate a correct and informative 
> response. The prompt should be designed to provide enough information to the 
> model to enable it to complete the task successfully, while avoiding unnecessary 
> details or biases that could lead to incorrect or unhelpful responses. 
> 
> https://stanford-cs324.github.io/winter2022/lectures/capabilities/
> 
> https://stanford-cs324.github.io/winter2022/lectures/modeling/
> 
> 
> Enter your Query: What is a tokenizer?
> 
> 
> In the context of transformers, a tokenizer is a function or algorithm 
> that takes a sequence of characters as input and outputs a sequence 
> of tokens. The tokens are typically unique symbols or subwords that represent 
> individual words or parts of words in the input sequence. The goal 
> of tokenization is to convert the raw sequence of characters into a 
> form that can be processed by a transformer model, which operates on 
> sequences of tokens rather than individual characters. 
> 
> https://stanford-cs324.github.io/winter2022/lectures/modeling/
> 
> 
> Enter your Query: Summarize your responses so far
> 
> 
> Sure! Transformers are a type of neural network architecture that has revolutionized 
> natural language processing tasks such as language translation, text generation, and question 
> answering. They are particularly useful for tasks that require long-range dependencies and 
> parallelization across input sequences. In transformer models, the input sequence is split 
> into multiple segments called "keys," "values," and "queries." The queries represent the 
> context in which the input sequence is being processed, while the keys 
> and values represent the input sequence itself. Attention weights are computed between 
> the queries and keys to determine how important each key is in 
> relation to the others. These attention weights are then used to compute 
> a weighted sum of the values, resulting in a transformed representation of 
> the input sequence. This process is repeated multiple times, with each iteration 
> refining the attention weights and transforming the input sequence further. Transformers have 
> several advantages over traditional recurrent neural network (RNN) architectures. They can process 
> input sequences of arbitrary length and do not suffer from the vanishing 
> gradient problem, which can limit the performance of RNNs. Additionally, transformers can 
> be parallelized more easily than RNNs, making them faster and more scalable 
> for large datasets. There are several variations of transformer models, including the 
> original transformer model introduced by Vaswani et al. (2017), the BERT model 
> (Devlin et al., 2018), and the RoBERTa model (Li et al., 2019). 
> Each of these models has its own strengths and weaknesses, but they 
> all share the basic transformer architecture. 
> 
> https://stanford-cs324.github.io/winter2022/lectures/modeling/
> 
> 
> Enter your Query: thanks

### References

[LLaMA 2 is here - get it on HuggingFace](https://huggingface.co/blog/llama2)

[FAISS](https://ai.meta.com/tools/faiss/)

[LangChain: Introduction and Getting Started](https://www.pinecone.io/learn/series/langchain/langchain-intro/)

[LLaMA 2: How to access and use Meta’s versatile open-source chatbot right now](https://venturebeat.com/ai/llama-2-how-to-access-and-use-metas-versatile-open-source-chatbot-right-now/)

[Faiss: A library for efficient similarity search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)

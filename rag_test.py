import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_path  = './models/'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, 
                                            # quantization_config=bnb_config, 
                                             device_map="auto")

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    return_full_text=True,
    repetition_penalty = 1.13,
    temperature = 0.1,
    # top_k = 40,
    max_new_tokens=500,
    do_sample = True
)

prompt_template = """아래 주어진 문맥에 해당하는 내용을 존댓말로 친절하게 답변해줘. 답변 안에 질문과 답변을 따로 하지마. 질문과 관련없는 답변은 하지마. 똑같은 문장을 의미없이 나열하지마.
    문맥 : {context}

 
    질문 : {question}
    답변 :"""

koplatyi_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

model_name = './models/ko-sbert-nli'
model_kwargs = {'device' : 'cuda'}
encode_kwargs = {'normalize_embeddings' : True}

hf = HuggingFaceEmbeddings(model_name = model_name,
                           model_kwargs = model_kwargs,
                           encode_kwargs = encode_kwargs
)

PERSIST_DIRECTORY = ''

DB_Chroma = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=hf)

retriever = DB_Chroma.as_retriever(
                            search_type="similarity",
                            search_kwargs={'k': 2}
)

qa = RetrievalQA.from_chain_type(
        llm=koplatyi_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

result = qa("질문")

print(result['result'])

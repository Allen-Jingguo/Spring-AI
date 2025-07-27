# serve_model.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

# 加载训练好的模型（根据保存位置调整路径）
model_path = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

class Query(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(query: Query):
    output = pipe(query.prompt, max_new_tokens=200, do_sample=True)
    return {"response": output[0]["generated_text"]}
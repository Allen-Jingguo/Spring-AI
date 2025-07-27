import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.llms import HuggingFacePipeline
from langchain.schema import OutputParserException
import dotenv
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import torch
import ssl
import threading
import re

def process_audio():
    while True:
        # 1. 录音
        print(f"🎤 开始录音（{duration} 秒）...")
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        write(audio_filename, samplerate, recording)
        print(f"✅ 录音完成，文件保存为：{audio_filename}")

        # 2. Whisper 本地识别
        print("🧠 正在加载 Whisper 模型并识别语音...")
        result = whisper_model.transcribe(audio_filename, language="en")
        recognized_text = result["text"]
        print("🎧 识别结果：", recognized_text)

        # 构造 Prompt
        system_prompt = "<|im_start|>system\n you are an AI assistant.<|im_end|>"
        user_prompt = f"<|im_start|>user\n{recognized_text}<|im_end|>\n<|im_start|>assistant\n"
        prompt = system_prompt + "\n" + user_prompt
        # query = input("\n请输入你的问题（输入 exit 退出）：\n> ")
        query = prompt
        pattern = r'\b(?:' + '|'.join(re.escape(word) for word in words) + r')\b'
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            print("👋 再见！")
            break
        print("\n🤖 处理查询：\n" + query)
        try:
            response = agent.invoke({"input": query, "chat_history": query})
            print("\n🤖 回答：\n" + str(response))
        except OutputParserException as e:
            print("\n❌ 解析错误：", str(e))
            print("⚠️ 请检查 Prompt 格式或代理配置。")


# 忽略 SSL 证书验证错误
ssl._create_default_https_context = ssl._create_unverified_context

# 参数设置
words = ["bye", "See you", "Goodbye"]
duration = 10  # 录音时长（秒）
samplerate = 16000  # Whisper 推荐采样率
audio_filename = "recorded.wav"

whisper_model = whisper.load_model("large-v3") # 或 "base", "small", "medium"

# ✅ 加载 .env 文件中的 API KEY
dotenv.load_dotenv()
assert os.getenv("SERPAPI_API_KEY"), "请先设置 SERPAPI_API_KEY 环境变量"

# ✅ Step 1: 加载 Qwen3-4B 模型
print("正在加载 Qwen3-4B 模型，请稍候...")
model_id = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")

# ✅ Step 2: 创建文本生成管道
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

# ✅ Step 3: 设置联网搜索工具（使用 SerpAPI）
search = SerpAPIWrapper()  # 使用环境变量中的 key
search_tool = Tool(
    name="web-search",
    func=search.run,
    description="适用于需要联网获取实时信息的问题，比如新闻、天气、股市等"
)

# ✅ Step 4: 初始化智能代理
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=False
)

# ✅ Step 5: 用户输入并执行查询
if __name__ == "__main__":
    print("\n欢迎使用 Qwen3-4B + 实时联网搜索代理")
    # 启动录音和处理线程
    thread = threading.Thread(target=process_audio)
    thread.start()

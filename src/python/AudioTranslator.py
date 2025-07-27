import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# 参数设置
duration = 10  # 录音时长（秒）
samplerate = 16000  # Whisper 推荐采样率
audio_filename = "recorded.wav"

# 1. 录音
print(f"🎤 开始录音（{duration} 秒）...")
recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
sd.wait()
write(audio_filename, samplerate, recording)
print(f"✅ 录音完成，文件保存为：{audio_filename}")

# 2. Whisper 本地识别
print("🧠 正在加载 Whisper 模型并识别语音...")
whisper_model = whisper.load_model("large-v3")  # 或 "base", "small", "medium"
result = whisper_model.transcribe(audio_filename, language="en")
recognized_text = result["text"]
print("🎧 识别结果：", recognized_text)

# 3. 使用 Qwen 本地生成回答
print("🤖 正在加载 Qwen 模型...")
# qwen_model_id = "Qwen/Qwen1.5-1.8B-Chat" Qwen3-8B
qwen_model_id = "Qwen/Qwen3-4B"  # 替换为你使用的 Qwen 模型 ID
tokenizer = AutoTokenizer.from_pretrained(qwen_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(qwen_model_id, trust_remote_code=True, device_map="auto").eval()

# 构造 Prompt
system_prompt = "<|im_start|>system\n你是一个语音助手。<|im_end|>"
user_prompt = f"<|im_start|>user\n{recognized_text}<|im_end|>\n<|im_start|>assistant\n"
prompt = system_prompt + "\n" + user_prompt

# 模型生成回答
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.8,
    top_p=0.9
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
if "<|im_start|>assistant\n" in response:
    response_text = response.split("<|im_start|>assistant\n")[-1]
else:
    response_text = response

print("🗣️ Qwen 回复：", response_text.strip())

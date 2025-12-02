import sherpa_onnx
from sherpa_onnx import OfflineRecognizer
from typing import Union
import librosa
import sounddevice as sd
import numpy as np
import requests
import time
import json


asr_path = 'model/ASR/sherpa-onnx-paraformer-zh-small-2024-03-09'
vad_path = 'model/VAD'

def get_command(text: str) -> list:
    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": "Qwen/Qwen3-8B",
        "messages": [
            {
                "role": "user",
                "content": f"""你是小车控制模块。请根据语音识别结果"{text}"，从以下动作集合中提取一个或多个**按顺序**要执行的动作，并以**JSON数组**格式返回。数组元素只能是这些动作：前进、后退、左转、右转、停止、加速、减速。每个动作的执行时间统一为5秒，返回结果不要包含其它文本或解释，只返回纯 JSON 数组，例如：["前进", "左转", "前进"]。

可选动作：前进、后退、左转、右转、无操作

匹配规则：
- 前进：前进、往前走、向前、直走等
- 后退：后退、往后退、倒车、向后等
- 左转：左转、向左转、左拐、往左等
- 右转：右转、向右转、右拐、往右等
- 无操作：停止、停下、或非控制指令

重要：只输出动作名称，不要添加任何其他文字。如果无法确定，输出"无操作"。

示例：
"前进" → 前进
"倒车" → 后退
"左拐" → 左转
"你好" → 无操作

回答："""
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "enable_thinking": False,
        "thinking_budget": 512,
        "min_p": 0.05,
        "stop": None,
        "temperature": 0.2,  # 降低温度以提高输出稳定性，适合分类任务
        "top_p": 0.3,  # 降低top_p以提高确定性
        "top_k": 10,  # 降低top_k，只考虑最可能的选项
        "frequency_penalty": 0.0,  # 移除频率惩罚，避免影响固定输出
        "n": 1,
        "response_format": {"type": "text"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "description": "<string>",
                    "name": "<string>",
                    "parameters": {},
                    "strict": False
                }
            }
        ]
    }
    headers = {
        "Authorization": "Bearer sk-diashhelzuktcatjjjvayunkrueyvchoapfhlnvdrwtsnocp",
        "Content-Type": "application/json"
    }
    try:
        response = requests.request("POST", url, json=payload, headers=headers)
        r_content = json.loads(response.text)['choices'][0]['message']['content'].strip()
        r_reason = json.loads(response.text)['choices'][0]['message']['reasoning_content'].strip()

        print('大模型回复：', r_content)
        print('大模型推理过程：', r_reason)
        # 尝试直接解析模型返回的 JSON 数组
        try:
            commands = json.loads(r_content)
            if isinstance(commands, list):
                # 仅保留合法命令
                legal_commands = [c for c in commands if isinstance(c, str) and c.strip() in ['前进', '后退', '左转', '右转', '停止', '加速', '减速']]
                return legal_commands
        except Exception:
            # 如果不是严格 JSON，则尝试从文本中抽取命令关键词顺序
            commands = []
            order_keywords = ['前进', '后退', '左转', '右转', '停止', '加速', '减速']
            for kw in order_keywords:
                # 按出现顺序查找，允许重复
                start = 0
                while True:
                    idx = r_content.find(kw, start)
                    if idx == -1:
                        break
                    commands.append(kw)
                    start = idx + len(kw)
            if len(commands) > 0:
                return commands
    except Exception as e:
        print('大模型请求失败：', e)
    return []

class ASR:
    def __init__(self):
        self._recognizer = OfflineRecognizer()
        raise NotImplementedError

    def transcribe(self, audio: Union[str, np.ndarray], sample_rate=16000) -> str:
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=sample_rate)
        s = self._recognizer.create_stream()
        s.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(s)
        return s.result.text


class Whisper(ASR):
    def __init__(self, encoder_path: str, decoder_path: str, tokens_path: str, num_threads: int = 8, provider: str = 'cpu'):
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=encoder_path,
            decoder=decoder_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=provider,
        )


class Paraformer(ASR):
    def __init__(self, model_path: str, tokens_path: str, num_threads: int = 8, provider: str = 'cpu'):
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=provider,
        )

print('正在加载ASR模型...')
asr = Paraformer(
    model_path=f'{asr_path}/model.int8.onnx',
    tokens_path=f'{asr_path}/tokens.txt',
    # provider='cuda',
)

sample_rate = 16000

from sherpa_onnx import VadModelConfig, SileroVadModelConfig, VoiceActivityDetector
config = VadModelConfig(
    SileroVadModelConfig(
        model=f'{vad_path}/silero_vad.onnx',
        min_silence_duration=0.25,
    ),
    sample_rate=sample_rate
)
window_size = config.silero_vad.window_size
vad = VoiceActivityDetector(config, buffer_size_in_seconds=100)
samples_per_read = int(0.1 * sample_rate)

control_url = "http://172.20.10.7:5000/control"  

# 速度控制参数 (0 ~ 100)
current_speed = 50
min_speed = 0
max_speed = 100
speed_step = 10
# 当前运动状态: 'STOP', 'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT'
movement_command = 'STOP'

def send_command(text):
    try:
        global current_speed, movement_command
        if '前进' == text:
            movement_command = 'FORWARD'
            response = requests.post(control_url, json={'command': "FORWARD", 'speed': current_speed})
        elif '后退' == text:
            movement_command = 'BACKWARD'
            response = requests.post(control_url, json={'command': "BACKWARD", 'speed': current_speed})
        elif '左转' == text:
            movement_command = 'LEFT'
            response = requests.post(control_url, json={'command': "LEFT", 'speed': current_speed})
        elif '右转' in text:
            movement_command = 'RIGHT'
            response = requests.post(control_url, json={'command': "RIGHT", 'speed': current_speed})
        elif '加速' in text or '快一点' in text or '快点' in text or '提速' in text:
            prev_speed = current_speed
            current_speed = min(max_speed, current_speed + speed_step)
            print(f'速度提升: {prev_speed} -> {current_speed}')
            if movement_command in ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']:
                response = requests.post(control_url, json={'command': movement_command, 'speed': current_speed})
            else:
                response = requests.post(control_url, json={'command': "SET_SPEED", 'speed': current_speed})
        elif '减速' in text or '慢一点' in text or '慢点' in text or '放慢' in text:
            prev_speed = current_speed
            current_speed = max(min_speed, current_speed - speed_step)
            print(f'速度下降: {prev_speed} -> {current_speed}')
            if movement_command in ['FORWARD', 'BACKWARD', 'LEFT', 'RIGHT']:
                response = requests.post(control_url, json={'command': movement_command, 'speed': current_speed})
            else:
                response = requests.post(control_url, json={'command': "SET_SPEED", 'speed': current_speed})
        else:
            movement_command = 'STOP'
            response = requests.post(control_url, json={'command': "STOP"})

        if response.status_code != 200:
            print('小车指令请求失败：', response)
    except Exception as e:
        print('小车指令请求异常：', e)

print('\n正在识别语音指令...')
idx = 1
buffer = []
try:
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)

            buffer = np.concatenate([buffer, samples])
            while len(buffer) > window_size:
                vad.accept_waveform(buffer[:window_size])
                buffer = buffer[window_size:]

            while not vad.empty():
                text = asr.transcribe(vad.front.samples, sample_rate=sample_rate)

                vad.pop()
                if len(text):
                    print()
                    print(f'第{idx}句：{text}')
                    commands = get_command(text)
                    if len(commands) == 0:
                        print('未识别到小车指令')
                    else:
                        print('识别到小车指令序列：', commands)
                        # 依次执行每个命令，每个命令执行 5s
                        for cmd in commands:
                            print('执行指令：', cmd)
                            send_command(cmd)
                            # 保持 5 秒
                            time.sleep(5)
                        # 执行序列完毕后发送停止命令
                        send_command('停止')
                    idx += 1
except KeyboardInterrupt:
    sd.stop()
    print('\n识别结束')

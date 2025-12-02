import sherpa_onnx
from sherpa_onnx import OfflineRecognizer
from typing import Union
import librosa
import sounddevice as sd
import numpy as np
import requests
import json


asr_path = 'model/ASR/sherpa-onnx-paraformer-zh-small-2024-03-09'
vad_path = 'model/VAD'

def get_command(text: str) -> str:
    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": "Qwen/Qwen3-8B",
        "messages": [
            {
                "role": "user",
                "content": f"""你是小车控制模块。根据语音识别结果"{text}"，从以下5个动作中选择一个并输出：

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
        response.raise_for_status()  # 检查HTTP状态码
        response_data = json.loads(response.text)
        r_content = response_data['choices'][0]['message']['content'].strip()
        
        # 只有当 enable_thinking 为 True 时才尝试获取推理过程
        r_reason = None
        if 'reasoning_content' in response_data['choices'][0]['message']:
            r_reason = response_data['choices'][0]['message']['reasoning_content'].strip()

        print('大模型回复：', r_content)
        if r_reason:
            print('大模型推理过程：', r_reason)
        command_list = ['前进', '后退', '左转', '右转', '无操作']
        rfind_idx_list = [
            r_content.rfind(command) for command in command_list
        ]
        max_idx = np.argmax(rfind_idx_list)
        if rfind_idx_list[max_idx] == -1:
            return '无操作'
        command = command_list[max_idx]
    except requests.exceptions.RequestException as e:
        print('大模型请求失败（网络错误）：', e)
        if hasattr(e, 'response') and e.response is not None:
            print('响应状态码：', e.response.status_code)
            print('响应内容：', e.response.text[:500])  # 只打印前500个字符
        command = '无操作'
    except KeyError as e:
        print('大模型请求失败（响应格式错误）：', e)
        print('响应内容：', response.text[:500] if 'response' in locals() else '无响应')
        command = '无操作'
    except json.JSONDecodeError as e:
        print('大模型请求失败（JSON解析错误）：', e)
        print('响应内容：', response.text[:500] if 'response' in locals() else '无响应')
        command = '无操作'
    except Exception as e:
        print('大模型请求失败（未知错误）：', e)
        print('错误类型：', type(e).__name__)
        command = '无操作'
    return command

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

def send_command(text):
    try:
        if '前进' == text:
            response = requests.post(control_url, json={'command': "FORWARD"})
        elif '后退' == text:
            response = requests.post(control_url, json={'command': "BACKWARD"})
        elif '左转' == text:
            response = requests.post(control_url, json={'command': "LEFT"})
        elif '右转' in text:
            response = requests.post(control_url, json={'command': "RIGHT"})
        else:
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
                    command = get_command(text)
                    if command == '无操作':
                        print('未识别到小车指令，发送停止命令')
                        send_command('停止')
                    else:
                        print('识别到小车指令：', command)
                        send_command(command)
                    idx += 1
except KeyboardInterrupt:
    sd.stop()
    print('\n识别结束')

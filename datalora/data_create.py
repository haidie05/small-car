import json, random, itertools, os

actions = ['前进','后退','左转','右转']
templates_single = [
    "{verb}","小车{verb}","请{verb}","现在{verb}吧","马上{verb}"
]
templates_seq_by_len = {
    2: [
        "先{0}再{1}",
        "请{0}，接着{1}",
        "{0}，随后{1}",
        "{0}，然后{1}"
    ],
    3: [
        "先{0}再{1}然后{2}",
        "请{0}，接着{1}，最后{2}",
        "小车{0}，{1}，再{2}",
        "{0}，随后{1}，接着{2}",
        "{0}，{1}，再{2}，最后停"
    ],
    4: [
        "先{0}再{1}随后{2}然后{3}",
        "请{0}，接着{1}，再{2}，最后{3}",
        "小车{0}，{1}，{2}，然后{3}",
    ],
    5: [
        "先{0}再{1}随后{2}然后{3}最后{4}",
        "请{0}，接着{1}，接着{2}，然后{3}，最后{4}",
    ]
}

def mk_single():
    a = random.choice(actions)
    t = random.choice(templates_single)
    sent = t.format(verb=a)
    # 输出格式改为 JSON 数组
    assistant_out = json.dumps([a], ensure_ascii=False)

    # Optionally add a system instruction (10% chance)
    messages = []
    if random.random() < 0.1:
        messages.append({
            'role': 'system',
            'content': '你是小车控制的语义解析模块。请仅按要求输出控制指令，不输出额外文字。'
        })
    messages.append({'role': 'user', 'content': sent})
    messages.append({'role': 'assistant', 'content': assistant_out})
    return {'messages': messages}

def mk_seq(n=3):
    # If you need more unique actions than we have, allow repeats (sample or choices)
    if n <= len(actions):
        acts = random.sample(actions, n)
    else:
        acts = random.choices(actions, k=n)
    # Choose a template that matches the number of placeholders
    tmpl_list = templates_seq_by_len.get(n)
    if tmpl_list is None:
        # Fallback to a simple comma separated sentence
        sent = '，'.join(acts)
    else:
        tmpl = random.choice(tmpl_list)
        sent = tmpl.format(*acts)
    
    # 输出格式改为简单的 JSON 数组，如 ["前进", "左转"]
    assistant_out = json.dumps(acts, ensure_ascii=False)
    messages = []
    if random.random() < 0.1:
        messages.append({
            'role': 'system',
            'content': '你是小车控制指令的解析模块。请输出 JSON 数组，数组元素为动作名称，数组顺序代表执行顺序。'
        })
    messages.append({'role': 'user', 'content': sent})
    messages.append({'role': 'assistant', 'content': assistant_out})
    return {'messages': messages}

# 生成数据，总计3000条
# 单条命令：800条
single = [mk_single() for _ in range(800)]
# 序列命令：2150条
seq2 = [mk_seq(2) for _ in range(600)]   # 2个动作的序列
seq3 = [mk_seq(3) for _ in range(850)]    # 3个动作的序列
seq4 = [mk_seq(4) for _ in range(500)]    # 4个动作的序列
seq5 = [mk_seq(5) for _ in range(200)]    # 5个动作的序列
# 停止命令：50条
stop = []
stop_words = ["停", "停下", "停止", "急停", "别动了", "停止运动", "停下来", "不要动了", "暂停", "停止前进",
              "停一下", "先停", "立即停止", "马上停", "快停", "停止运行", "停止移动", "停止操作", "停止前进",
              "停止后退", "停止左转", "停止右转", "停止所有动作", "停止一切", "停止执行", "停止工作",
              "停止运行", "停止移动", "停止前进", "停止后退", "停止左转", "停止右转", "停止所有动作",
              "停止一切", "停止执行", "停止工作", "停止运行", "停止移动", "停止前进", "停止后退",
              "停止左转", "停止右转", "停止所有动作", "停止一切", "停止执行", "停止工作"]
for w in stop_words[:50]:  # 取前50个
    assistant_out = json.dumps(["停止"], ensure_ascii=False)
    messages = [{'role': 'user', 'content': w}, {'role': 'assistant', 'content': assistant_out}]
    stop.append({'messages': messages})

all_data = single + seq2 + seq3 + seq4 + seq5 + stop
random.shuffle(all_data)
out_path = os.path.join(os.path.dirname(__file__), 'raspberry_car_3000.jsonl')
with open(out_path, 'w', encoding='utf-8') as f:
    for d in all_data:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')
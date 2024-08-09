#!/usr/bin/env python3
import time  # 导入时间模块
import termcolor  # 导入终端颜色模块
from fastapi import FastAPI, UploadFile, File  # 从FastAPI导入FastAPI类、UploadFile和File
from pydantic import BaseModel  # 从Pydantic导入BaseModel
from typing import List  # 从typing模块导入List
from nano_llm import NanoLLM, ChatHistory  # 从nano_llm导入NanoLLM和ChatHistory
from nano_llm.utils import ArgParser, load_prompts  # 从nano_llm.utils导入ArgParser和load_prompts
from nano_llm.plugins import VideoSource  # 从nano_llm.plugins导入VideoSource
from jetson_utils import cudaMemcpy, cudaToNumpy  # 从jetson_utils导入cudaMemcpy和cudaToNumpy

# 定义请求模型
class RequestModel(BaseModel):
    prompts: List[str]  # 提示列表
    model: str = "Efficient-Large-Model/VILA1.5-3b"  # 模型名称，默认值
    video_input: str = "/data/images/*.jpg"  # 视频输入路径，默认值

# 创建FastAPI应用实例
app = FastAPI()

# 定义全局变量用于模型和聊天历史记录
model = None
chat_history = None
video_source = None

# 定义应用的生命周期事件
@app.on_event("startup")
async def startup_event():
    global model, chat_history, video_source  # 使用全局变量

    # 解析参数并设置一些默认值
    args = ArgParser(extras=ArgParser.Defaults + ['prompt', 'video_input']).parse_args()
    prompts = load_prompts(args.prompt)  # 加载提示

    if not prompts:  # 如果提示为空，则使用默认提示
        prompts = ["Describe the image.", "Are there people in the image?"]

    if not args.model:  # 如果模型参数为空，则使用默认模型
        args.model = "Efficient-Large-Model/VILA1.5-3b"

    if not args.video_input:  # 如果视频输入参数为空，则使用默认视频输入路径
        args.video_input = "/data/images/*.jpg"

    # 加载视觉/语言模型
    model = NanoLLM.from_pretrained(
        args.model,
        api=args.api,
        quantization=args.quantization,
        max_context_len=args.max_context_len,
        vision_model=args.vision_model,
        vision_scaling=args.vision_scaling,
    )

    assert(model.has_vision)  # 确保模型具有视觉功能

    # 创建聊天历史记录
    chat_history = ChatHistory(model, args.chat_template, args.system_prompt)

    # 打开视频流
    video_source = VideoSource(**vars(args), cuda_stream=0, return_copy=False)

# 定义处理图像请求的端点
@app.post("/process_image/")
async def process_image(request: RequestModel):
    global chat_history, model, video_source  # 使用全局变量

    # 更新提示和模型（如果需要）
    prompts = request.prompts

    try:
        # 尝试从视频源捕获图像
        img = video_source.capture()
        if img is None:  # 如果图像为空，返回错误信息
            return {"error": "Failed to capture image"}
    except Exception as e:
        # 捕获异常并返回错误信息
        return {"error": str(e)}

    chat_history.append('user', image=img)  # 将图像添加到聊天历史记录
    time_begin = time.perf_counter()  # 记录开始时间

    results = []  # 初始化结果列表

    for prompt in prompts:  # 遍历每个提示
        chat_history.append('user', prompt, use_cache=True)  # 将提示添加到聊天历史记录
        embedding, _ = chat_history.embed_chat()  # 嵌入聊天

        print('>>', prompt)  # 打印提示

        # 生成回复
        reply = model.generate(
            embedding,
            kv_cache=chat_history.kv_cache,
            max_new_tokens=50,  # 最大新标记数
            min_new_tokens=1,  # 最小新标记数
            do_sample=True,
            repetition_penalty=1.0,
            temperature=1.0,
            top_p=0.9,
            streaming=False,
        )
        result_text = reply

        print('\nresult_text:', result_text)  # 打印提示
        # result_text = ''  # 初始化结果文本
        # for token in reply:  # 遍历回复中的每个标记
        #     result_text += token  # 将标记添加到结果文本
        #     termcolor.cprint(token, 'blue', end='\n\n' if reply.eos else '', flush=True)  # 以蓝色打印标记

        chat_history.append('bot', reply)  # 将回复添加到聊天历史记录
        results.append(result_text)  # 将结果文本添加到结果列表

    time_elapsed = time.perf_counter() - time_begin  # 计算经过的时间
    print(f"time: {time_elapsed * 1000:.2f} ms rate: {1.0 / time_elapsed:.2f} FPS")  # 打印时间和帧率

    chat_history.reset()  # 重置聊天历史记录

    # 返回结果、经过时间和帧率
    return {"results": results, "time_elapsed_ms": time_elapsed * 1000, "fps": 1.0 / time_elapsed}

# 运行应用
if __name__ == "__main__":
    import uvicorn  # 导入uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # 运行FastAPI应用

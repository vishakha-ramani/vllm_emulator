## https://towardsdatascience.com/how-to-build-an-openai-compatible-api-87c8edea2f06

import time
import datetime
import threading

from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel

import asyncio

############################################  SETUP vLLM Emulator #####################################################

from vllm_model import *


clock = Clock(start_time = 0, step_time = DECODE_TIME)
model = Model(model_id = 1, model_size = 25000, kvcache_per_token = KVC_PER_TOKEN)
gpu   = Device(device_id = 1, net_memory = M, useable_ratio = 0.8)

vllmi = vLLM( device=gpu, clock=clock, model=model)
load  = Load( avg_generated_len = 100, distribution = 'uniform')

## Start as vLLM
## Note: vllm emulator's run() is currently just a while loop. So not mkaing it a thread will deadlock the server.py
vllm_thread = threading.Thread(target=vllmi.run)
vllm_thread.start()
## TODO: chat_completions() modifies the finished_queue. Havent checked thread safety of this setup

######################################################################################################################


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mock-gpt-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

app = FastAPI(title="OpenAI-compatible API")

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):

    ## https://stackoverflow.com/questions/76913406/how-allow-fastapi-to-handle-multiple-requests-at-the-same-time
    ## Async can't have simple sleep function, else will block the server
    async def sleep_async(rid):
        await asyncio.sleep(0.2)
        print(f"Waiting for completion {rid}")
    

    input_seq = request.messages[-1].content

    input_len  = len(input_seq)
    output_len = load.get_output_len(input_len)
    now    = datetime.datetime.now()
    req_id = now.strftime("%Y-%m-%dT%H:%M:%S-") + str(random.randint(0,100))

    reqi   = RequestElement(req_id=req_id, input_token_length=input_len, output_token_length=output_len)
    vllmi.add_new_request(reqi)


    complete = False
    while not complete:
       print(f"Waiting for completion {req_id}")
       complete = vllmi.remove_finished_request(reqi)
       await asyncio.sleep(0.5)                               #TODO: Sleep spinning to check very bad! Replace

    
    if request.messages and request.messages[0].role == 'user':
      resp_content = f"Request stats: arrival time = {reqi.arrival_time}, completion time = {reqi.completion_time}, ttft = {reqi.ttft_met_time}, input_token_len = {reqi.InputTokenLength}, output_token_len = {reqi.token_len}"
    else:
      resp_content = "Empty message sent!"

    
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{
            "message": ChatMessage(role="assistant", content=resp_content)
        }]
    }




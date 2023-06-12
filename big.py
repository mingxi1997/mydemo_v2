"""
Use FastChat with Hugging Face generation APIs.
Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
python3 -m fastchat.serve.huggingface_api --model ~/model_weights/vicuna-7b/
"""
import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.model import load_model, get_conversation_template, add_model_args



parser = argparse.ArgumentParser()
add_model_args(parser)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--debug", action="store_true")
# parser.add_argument("--message", type=str, default="Hello! Who are you?")
args = parser.parse_args()

args.model_path='/home/slkj/Chinese-LLaMA-Alpaca-Plus-13b-merge'
model, tokenizer = load_model(
    args.model_path,
    args.device,
    args.num_gpus,
    args.max_gpu_memory,
    args.load_8bit,
    args.cpu_offloading,
    debug=args.debug,
)

def inference(ask):


    with torch.no_grad():
    
        input_ids = tokenizer([ask]).input_ids
        
      
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )
    
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return outputs


msg='您好，请介绍下唐朝的历史'

conv = get_conversation_template(args.model_path)
conv.append_message(conv.roles[0], msg)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

prompt="A chat between a curious human and an artificial intelligence assistant. \
    The assistant gives helpful, detailed, and polite answers to the human's questions."
    
ask=prompt+'\n### Human: '+msg+'\n### Assistant:唐朝是中国历史上一个非常重要的朝代，它从公元618年到公元907年存在，\
    是中国文化和科技的鼎盛时期。唐朝的文化和科技在当时被誉为“世界文化的明珠”，其诗歌、绘画、音乐、舞蹈、戏剧等艺术形式，\
        被誉为中国古代艺术的巅峰。'+'\n### Human: 之后是哪个朝代 \n### Assistant:'
output=inference(ask)

print(output)


import os
from typing import Optional

import torch
import backoff
from openai import OpenAI
from huggingface_hub import hf_hub_download, login
from transformers import GPT2Tokenizer
#from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from ctransformers import AutoModelForCausalLM as qAutoModelForCausalLM 
from ctransformers import AutoConfig as qAutoConfig

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

default_openai_key = os.getenv("OPENAI_API_KEY")

class GPTRunner():
    def __init__(self, debug: bool = False):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.max_new_tokens = 400
        self.__debug = debug

    def run(self, prompt: str, model: str="gpt-3.5-turbo-1106") -> Optional[str]:
        print('Input prompt for llm:')
        print(prompt)
        print('prompt end')
        result = self.__call_gpt(prompt, model)
        print(result)
        if self.__debug:
            with open("debug.txt", "a", encoding="utf8") as w:
                w.write("#" * 10)
                w.write(" EXAMPLE ")
                w.write("#" * 10)
                w.write("\n")
                w.write("### PROMPT ###\n")
                w.write(prompt)
                w.write("\n\n")
                w.write("### RESULT ###\n")
                w.write(result)
                w.write("\n\n")
        return result

    def is_prompt_too_long(self, prompt: str) -> bool:
        """Returns whether a given prompt is too long after tokenization."""
        tokens = self.tokenizer(prompt)

        print(len(tokens["input_ids"]) + self.max_new_tokens)
        if len(tokens["input_ids"]) + self.max_new_tokens > 4097:
            return True
        else:
            return False

    def __call_gpt(self, prompt: str, model: str) -> str:
        api_result = self.__api_call(prompt=prompt, model=model)
        return api_result.choices[0].message.content

    @backoff.on_exception(backoff.expo, Exception)
    def __api_call(self, prompt: str, model: str):
        client = OpenAI()
        print(client)
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            frequency_penalty=0.0,
            max_tokens=self.max_new_tokens,
            presence_penalty=0.6,
            n=1,
            temperature=0.0,
            stream=False,
)
    
class LLMRunner():
    def __init__(self, model:str, model_type:str, LORAConfig, gpu:bool = False, qlora=True, gpu_index=0):
        self.gpu_index = gpu_index
        self.model_type = model_type
        self.model_path = load_llm(model, model_type)
        self.config = self.reset_config(self.model_path, model_type, gpu)
        if model_type=='4bit':
            #nf4_config = BitsAndBytesConfig(
                                           #load_in_4bit=True,
                                           #bnb_4bit_quant_type="nf4",
                                           #bnb_4bit_use_double_quant=True,
                                           #bnb_4bit_compute_dtype=torch.bfloat16
                                           #)
            if model!='phi': 
                self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.model_path, config=self.config, 
                                                                 trust_remote_code=True, low_cpu_mem_usage=True,
                                                                 device_map=f"cuda:{gpu_index}", load_in_4bit=True)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.model_path,
                                                                 trust_remote_code=True, low_cpu_mem_usage=True,
                                                                 device_map=f"cuda:{gpu_index}", load_in_4bit=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        elif model_type=='full':
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.model_path, config=self.config, 
                                                              trust_remote_code=True, device_map=f"cuda:{gpu_index}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            self.model = qAutoModelForCausalLM.from_pretrained(model_path_or_repo_id=self.model_path, config=self.config)
        if qlora and model_type!='quant':
            lora_config = LoraConfig(
                                    r=LORAConfig.r,
                                    lora_alpha=LORAConfig.lora_alpha,
                                    target_modules=['q_proj','k_proj','v_proj','o_proj'],
                                    lora_dropout=LORAConfig.lora_dropout,
                                    bias="none",
                                    task_type=TaskType.SEQ_CLS, 
                                    )
            self.model = get_peft_model(prepare_model_for_kbit_training(self.model), lora_config)
        self.max_new_tokens = 400

    def is_prompt_too_long(self, prompt: str) -> bool:
        pass

    def reset_config(self, model_path, model_type, gpu):
        if model_type=='quant':
            config = qAutoConfig.from_pretrained(model_path_or_repo_id=model_path)
            if gpu:
                config.config.gpu_layers = 36
            config.config.context_length = 4096
            config.config.temperature = 0.0
            config.config.repetition_penalty = 1.1
            config.config.top_p = 1
        else: 
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path)
            config.context_length = 4096
            config.temperature = 0.0
            config.repetition_penalty = 1.1
            config.top_p = 1
        return config
    
    @property
    def model_config(self):
        return self.model.config
    
    @model_config.setter
    def model_config(self, config): 
        self.model.config = config  

    def run(self, prompt: str) -> Optional[str]:
        if self.model_type=="quant":
            result = self.__call__llm__quant(prompt)
        else:
            result = self.__call__llm__full(prompt)
        return result

    def __call__llm__quant(self, prompt: str) -> str:
        prompt = prompt[:self.model.config.context_length]
        result = self.model.__call__(
            prompt=prompt,
            temperature=self.config.config.temperature,
            top_p=self.config.config.top_p,
            max_new_tokens=self.max_new_tokens
        )
        return result
    
    def __call__llm__full(self, prompt: str) -> str:
        tokens = self.tokenizer(prompt+'[/INST]', return_tensors='pt').to(f'cuda:{self.gpu_index}')
        result = self.model.generate(tokens['input_ids'], 
                                     max_new_tokens = self.max_new_tokens, 
                                     do_sample = False, 
                                     repetition_penalty = self.config.repetition_penalty, 
                                     top_p=self.config.top_p)[0]
        result = self.tokenizer.decode(result, skip_special_tokens=True).split('[/INST]')[1]
        return result   
    
def load_llm(model:str, model_type:str):
    if model_type=="quant":
        files = os.listdir(f'iglu_datasets/data/models/{model}')
        model_name = []
        for file in files:
            if file.endswith(".gguf"):
                model_name.append(file)
        if not model_name:
            if model == 'mistral':
                hf_hub_download(repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                                filename="mistral-7b-instruct-v0.1.Q5_K_M.gguf", 
                                local_dir='iglu_datasets\data\models\mistral', force_download=True)
                model_name = "mistral-7b-instruct-v0.1.Q5_K_M.gguf"
            elif model == 'mixtral':
                hf_hub_download(repo_id="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
                                filename="mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf", 
                                local_dir=os.getcwd()+'\iglu_datasets\data\models\mixtral', force_download=True)
                model_name = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"   
            elif model == 'llama':
                hf_hub_download(repo_id="TheBloke/Llama-2-13B-GGUF",
                                filename="llama-2-13b.Q5_K_M.gguf", 
                                local_dir='iglu_datasets\data\models\llama', force_download=True)
                model_name = "llama-2-13b.Q5_K_M.gguf"
            elif model == 'solar':
                hf_hub_download(repo_id="TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF",
                                filename="solar-10.7b-instruct-v1.0.Q5_K_M.gguf", 
                                local_dir='iglu_datasets\data\models\solar', force_download=True)
                model_name = "solar-10.7b-instruct-v1.0.Q5_K_M.gguf"
            elif model == 'phi':
                hf_hub_download(repo_id="QuantFactory/Phi-3-mini-128k-instruct-GGUF",
                                filename="Phi-3-mini-128k-instruct.Q5_K_M.gguf", 
                                local_dir='iglu_datasets\data\models\phi', force_download=True)
                model_name = "Phi-3-mini-128k-instruct.Q5_K_M.gguf"     
        else:
            model_name = model_name[0]
        model_path = os.path.join(os.getcwd(), f'iglu_datasets/data/models/{model}', model_name)
    else:
        if model == 'mistral':
            model_path = "mistralai/Mistral-7B-Instruct-v0.2"
        elif model == 'mixtral':
            login('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')
            model_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        elif model == 'llama':
            model_path = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
        elif model == 'solar':
            model_path = "upstage/SOLAR-10.7B-Instruct-v1.0"
        elif model == 'phi':
            model_path = "unsloth/Phi-3-mini-4k-instruct"
    return model_path 
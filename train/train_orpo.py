# Copyright 2024 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import shutil
import logging
from dataclasses import dataclass, field
import deepspeed
from typing import Optional, Dict

from datasets import load_dataset
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
from trl import ORPOConfig, ORPOTrainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.modeling_utils import unwrap_model


def print_args(args, name='arguments'):
    """Print arguments."""
    if torch.distributed.get_rank() == 0:
        print(f'------------------------ {name} ------------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {name} ---------------------', flush=True)


@dataclass
class ModelArguments:
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )
    use_lora: bool = field(default=False, metadata={"help": "Enable Lora for faster training."})
    hidden_size: int = field(default=2048, metadata={"help": "The hidden size of the model."})
    num_layers: int = field(default=24, metadata={"help": "The number of layers of the model."})
    num_attention_heads: int = field(default=16, metadata={"help": "The number of attention heads of the model."})
    intermediate_size: int = field(default=8192, metadata={"help": "The intermediate size of the model."})
    max_position_embeddings: int = field(
        default=2048,
        metadata={"help": "The maximum sequence length that this model might ever be used with."}
    )
    vocab_size: int = field(default=50257, metadata={"help": "The vocabulary size of the model."})
    type_vocab_size: int = field(default=1, metadata={"help": "The vocabulary size of the model."})
    layer_norm_eps: float = field(
        default=1e-5,
        metadata={"help": "The epsilon used by the layer normalization layers of the model."}
    )
    moe_topk: int = field(default=4, metadata={"help": "The topk for MOE."})
    num_experts: int = field(default=8, metadata={"help": "The number of experts for MOE."})
    num_key_value_heads: int = field(default=16, metadata={"help": "The number of key-value heads in GQA."})
    use_cla: bool = field(default=False, metadata={"help": "Whether to use CLA."})
    cla_share_factor: int = field(default=2, metadata={"help": "The share factor for CLA."})
    use_mixed_mlp_moe: bool = field(
        default=False,
        metadata={"help": "Whether to use mixed MoE with shared expert."}
    )
    num_shared_expert: int = field(default=1, metadata={"help": "Number of shared experts."})
    use_qk_norm: bool = field(default=False, metadata={"help": "Whether to use qk norm."})
    tie_word_embeddings: bool = field(
        default=True,
        metadata={"help": "Whether to tie the word embeddings of the encoder and the decoder."}
    )
    lora_rank: int = field(default=64, metadata={"help": "The rank of lora."})
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    train_attention_params_only: bool = field(default=False, metadata={
        "help": "Whether to train attention parameters only."}
                                              )


@dataclass
class DataArguments:
    train_data: str = field(default=None, metadata={"help": "HF dataset name."})
    train_split: str = field(default="train", metadata={"help": "HF dataset split."})
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The max sequence length of the model inputs after tokenization."}
    )
    complex_data: Optional[str] = field(default=None)
    use_dummy_data: bool = field(default=False, metadata={"help": "Use dummy data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    hf_token: str = field(default=None)
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    tokenizer_name_or_path: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default=None)
    make_moe_param_leaf_module: bool = field(
        default=False,
        metadata={"help": "Make MoE parameters zero-3 leaf module."}
    )
    min_lr: float = field(
        default=0.01,
        metadata={"help": "The final learning rate at the end of the decay will be learning_rate * min_lr"}
    )


IGNORE_INDEX = -100


# full 模型训练时，需要修改 config.json 以及拷贝模型与配置文件支持 Auto load
class CustomSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if torch.distributed.get_rank() == 0:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            # 拷贝tokenizer, 模型和配置文件
            model_path = os.path.join(args.model_name_or_path, 'modeling_hunyuan.py')
            config_path = os.path.join(args.model_name_or_path, 'configuration_hunyuan.py')
            shutil.copy(model_path, os.path.join(output_dir, 'modeling_hunyuan.py'))
            shutil.copy(config_path, os.path.join(output_dir, 'configuration_hunyuan.py'))
            shutil.copy(
                os.path.join(args.tokenizer_name_or_path, 'generation_config.json'),
                os.path.join(output_dir, 'generation_config.json')
            )
            shutil.copy(
                os.path.join(args.tokenizer_name_or_path, 'hy.tiktoken'),
                os.path.join(output_dir, 'hy.tiktoken')
            )
            shutil.copy(
                os.path.join(args.tokenizer_name_or_path, 'tokenizer_config.json'),
                os.path.join(output_dir, 'tokenizer_config.json')
            )
            shutil.copy(
                os.path.join(args.tokenizer_name_or_path, 'tokenization_hy.py'),
                os.path.join(output_dir, 'tokenization_hy.py')
            )

            # 修改 config.json，增加 auto_map
            if os.path.exists(os.path.join(output_dir, "config.json")):
                config = json.load(open(os.path.join(output_dir, "config.json"), 'r'))
                config['auto_map'] = {
                    "AutoConfig": "configuration_hunyuan.HunYuanConfig",
                    "AutoModel": "modeling_hunyuan.HunyuanModel",
                    "AutoModelForCausalLM": "modeling_hunyuan.HunYuanForCausalLM"
                }
                json.dump(config, open(os.path.join(output_dir, "config.json"), 'w'), indent=2)

        return control


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print_args(model_args, 'model arguments')
    print_args(data_args, 'data arguments')
    print_args(training_args, 'training arguments')

    train_data = load_dataset(data_args.train_data, split=data_args.train_split, token=training_args.hf_token)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_args.tokenizer_name_or_path,
        trust_remote_code=True
    )

    init_kwargs = {}
    if model_args.use_flash_attn:
        init_kwargs["attn_implementation"] = "flash_attention_2"
    if training_args.bf16:
        init_kwargs["torch_dtype"] = torch.bfloat16
    elif training_args.fp16:
        init_kwargs["torch_dtype"] = torch.float16

    if training_args.model_name_or_path is not None and os.path.exists(training_args.model_name_or_path):
        print(f"Initializing model from local file: {training_args.model_name_or_path}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            training_args.model_name_or_path,
            trust_remote_code=True,
            **init_kwargs
        )
    else:
        from models.modeling_hunyuan import HunYuanForCausalLM, HunYuanMoE
        from models.configuration_hunyuan import HunYuanConfig
        print(f"Model name or path does not exist: {training_args.model_name_or_path}, \
              use random initialized model instead.")
        # 定义模型
        config = HunYuanConfig(
            vocab_size=tokenizer.vocab_size,  # 词表大小
            hidden_size=model_args.hidden_size,  # 隐藏层大小
            intermediate_size=model_args.intermediate_size,  # FFN 层大小
            max_position_embeddings=training_args.model_max_length,  # 最大序列长度
            moe_topk=model_args.moe_topk,  # topk
            num_experts=model_args.num_experts,  # expert 数量
            num_attention_heads=model_args.num_attention_heads,  # 多头注意力头数
            num_key_value_heads=model_args.num_key_value_heads,  # GQA 时的 key value 头数
            num_hidden_layers=model_args.num_layers,  # Transformer 层数
            cla_share_factor=model_args.cla_share_factor,  # CLA 因子
            use_cla=model_args.use_cla,
            use_mixed_mlp_moe=model_args.use_mixed_mlp_moe,
            num_shared_expert=model_args.num_shared_expert,
            use_qk_norm=model_args.use_qk_norm,
            model_type='hunyuan',
            tie_word_embeddings=model_args.tie_word_embeddings,
            **init_kwargs
        )
        with deepspeed.zero.Init(dtype=init_kwargs["torch_dtype"], config_dict_or_path=training_args.deepspeed):
            model = HunYuanForCausalLM(config)

    if model_args.train_attention_params_only:
        for name, param in model.named_parameters():
            if 'self_attn' not in name:
                param.requires_grad = False

    if model_args.use_lora:
        # 定义 Lora 配置
        peft_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # 用 zero3 的时候不切分 MoE 参数
    if model_args.num_experts > 0 and training_args.make_moe_param_leaf_module and training_args.deepspeed_plugin.zero_stage == 3:
        from deepspeed.utils import set_z3_leaf_modules
        set_z3_leaf_modules(model, [HunYuanMoE])

    # Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    training_args.lr_scheduler_kwargs = {'min_lr': training_args.min_lr}
    training_args.max_length = training_args.model_max_length
    training_args.report_to = ["comet_ml"]

    training_kwargs = training_args.to_dict()
    del training_kwargs["cache_dir"]
    del training_kwargs["hf_token"]
    del training_kwargs["model_name_or_path"]
    del training_kwargs["tokenizer_name_or_path"]
    del training_kwargs["model_max_length"]
    del training_kwargs["make_moe_param_leaf_module"]
    del training_kwargs["min_lr"]
    orpo_args = ORPOConfig(**training_kwargs)

    trainer = ORPOTrainer(
        model=model,
        train_dataset=train_data,
        tokenizer=tokenizer,
        args=orpo_args,
        peft_config=peft_config,
        callbacks=[CustomSaveCallback],
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == "__main__":
    train()

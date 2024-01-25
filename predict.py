import gc
from threading import Thread
from hqq.core.quantize import BaseQuantizeConfig
from transformers import AutoConfig, AutoTokenizer
from src.build_model import OffloadConfig, QuantConfig, build_model
import os
import torch
from cog import BasePredictor, Input, ConcatenateIterator
from utils import maybe_download_with_pget, delay_prints

# def delay_prints(REALLY_EAT_MY_PRINT_STATEMENTS: bool = False) -> tp.Iterator[tp.Callable]:


# Set HF_HOME before importing transformers
CACHE_DIR = "./hf-cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from transformers import (  # noqa: E402
    AutoTokenizer,
    TextIteratorStreamer,
)

PROMPT_TEMPLATE = "<s>[INST] {prompt} [/INST] "


DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_PRESENCE_PENALTY = 0.0  # 1.15
DEFAULT_FREQUENCY_PENALTY = 0.0  # 0.2


class Predictor(BasePredictor):
    """inference server for mixtral"""

    def setup(self):
        # download model using pget
        # get list of files from manifest
        remote_filenames = []

        with open("MANIFEST.txt", "r") as manifest_file:
            for line in manifest_file:
                remote_filenames.append(line.strip())
        print(remote_filenames)
        #print(remote_filenames)
        maybe_download_with_pget(
            path=os.path.join("models", "Mixtral-8x7B-Instruct-v0.1-offloading-demo"),
            remote_path="https://weights.replicate.delivery/wqzt/08ef4cd6-1975-49a3-b023-972c2e2cb95f/Mixtral-8x7B-Instruct-v0.1-offloading-demo",
            remote_filenames=remote_filenames,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixtral_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.quantized_mixtral_model_name = (
            "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
        )
        self.state_path = "models/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
        self.config = AutoConfig.from_pretrained(self.quantized_mixtral_model_name)
        self.past_key_values = None
        ##### Change this to 5 if you have only 12 GB of GPU VRAM #####
        self.offload_per_layer = 0
        # offload_per_layer = 5
        ###############################################################
        self.num_experts = self.config.num_local_experts

        self.offload_config = OffloadConfig(
            main_size=self.config.num_hidden_layers
            * (self.num_experts - self.offload_per_layer),
            offload_size=self.config.num_hidden_layers * self.offload_per_layer,
            buffer_size=4,
            offload_per_layer=self.offload_per_layer,
        )

        self.attn_config = BaseQuantizeConfig(
            nbits=4,
            group_size=64,
            quant_zero=True,
            quant_scale=True,
        )
        self.attn_config["scale_quant_params"]["group_size"] = 256

        self.ffn_config = BaseQuantizeConfig(
            nbits=2,
            group_size=16,
            quant_zero=True,
            quant_scale=True,
        )
        self.quant_config = QuantConfig(
            ffn_config=self.ffn_config, attn_config=self.attn_config
        )

        self.mixtral_model = build_model(
            device=self.device,
            quant_config=self.quant_config,
            offload_config=self.offload_config,
            state_path=self.state_path,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.mixtral_model_name)
        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

    @delay_prints(REALLY_EAT_MY_PRINT_STATEMENTS=True)
    def predict(
        self,
        prompt: str,
        max_new_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            default=DEFAULT_MAX_NEW_TOKENS,
        ),
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.",
            default=DEFAULT_TEMPERATURE,
        ),
        top_p: float = Input(
            description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
            default=DEFAULT_TOP_P,
        ),
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
            default=DEFAULT_TOP_K,
        ),
    ) -> ConcatenateIterator:
        """infer function"""
        print(f"=== Formatted Prompt ===\n{prompt}\n{'=' * 24}\n")
        # get response from LLM
        user_entry = dict(role="user", content=prompt)
        input_ids = self.tokenizer.apply_chat_template(
            [user_entry], return_tensors="pt"
        ).to(self.device)
        if self.past_key_values is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            seq_len = input_ids.size(1) + self.past_key_values[0][0][0].size(1)
            attention_mask = torch.ones(
                [1, seq_len - 1], dtype=torch.int, device=self.device
            )
        generate_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            do_sample=True,
            temperature=temperature,
            streamer=self.streamer,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=False,
            output_hidden_states=False,
        )
        t = Thread(target=self.mixtral_model.generate, kwargs=generate_kwargs)
        t.start()
        for text in self.streamer:
            yield text
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    p = Predictor()
    p.setup()
    for text in p.predict(
        "What time is the game tomorrow?",
    ):
        print(text, end="")

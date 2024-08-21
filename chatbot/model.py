import torch
from mlx_lm import convert, load
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from chatbot.constants import CPU, CUDA, LLM_MODEL, MACOS, MPS
from chatbot.utils import get_os


class ModelLoader:
    os_type = get_os()

    def __init__(self) -> None:
        raise EnvironmentError(
            "ModelLoader is designed to be instantiated "
            "using the `ModelLoader.load(model_name_or_path)` method."
        )

    @classmethod
    def load(cls, model_name_or_path: str, quantize: bool) -> tuple:
        device = (
            torch.device(MPS if cls.os_type == MACOS else CUDA)
            if torch.cuda.is_available()
            else torch.device(CPU)
        )
        if cls.os_type == MACOS:
            if quantize:
                convert(model_name_or_path, quantize=True)
            llm, tokenizer = load(
                model_name_or_path,
                tokenizer_config={
                    "use_fast": True,
                    "add_eos_token": False,
                    "clean_up_tokenization_spaces": True,
                },
                model_config={"return_full_text": True, "device_map": device},
            )
        else:
            quantization_config = None
            if quantize:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=False,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            llm = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=device,
                quantization_config=quantization_config,
            )
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)

        return llm, tokenizer

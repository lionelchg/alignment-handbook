import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Optional
from colored import Fore, Style

def run_chat(
        model_name: str="mistralai/Mistral-7B-Instruct-v0.1",
        device: str="cuda",
        system_message: Optional[str]=None,
        end_user_message: str="[/INST]",
    ) -> None:
    # Load parameters
    print((f"{Fore.green}Loading {model_name} on device {device} {Style.reset}"))
    print(f"{Fore.yellow}Using following system message:{Style.reset}\n{system_message}")
    print(f"{Fore.yellow}Capturing end of message with:{Style.reset} {end_user_message}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.sep_token = tokenizer.eos_token
    # tokenizer.cls_token = tokenizer.eos_token
    # tokenizer.mask_token = tokenizer.eos_token

    # Generation parameters
    temperature = 0
    top_p = 1.0
    top_k = 50
    repetition_penalty = 1.0 # no penalty
    max_new_tokens = 400
    generation_cfg = GenerationConfig(
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    messages = []
    if system_message is not None:
        messages.append(
            {"role": "system", "content": system_message}
        )
    while True:
        user_message = input("\033[1;32mUser\033[0m: ")
        messages.append(
            {"role": "user", "content": user_message}
        )
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)

        generated_ids = model.generate(
            model_inputs,
            generation_config=generation_cfg
        )
        decoded = tokenizer.batch_decode(generated_ids)
        output = decoded[0]

        # Search from the last end_user_message
        answer = output[output.rfind(end_user_message) + len(end_user_message):].strip()
        answer = answer[:len(answer) - len(tokenizer.eos_token)]
        print("\033[1;33mAssistant\033[0m: ", answer)

        messages.append(
            {"role": "assistant", "content": answer}
        )

if __name__ == "__main__":
    fire.Fire(run_chat)
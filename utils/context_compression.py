COMPRESSION_PROMPT = """
<bos><start_of_turn>user
Extract and summarize the information and description of each products in the content.

content:
{context}<end_of_turn>
<start_of_turn>model
""".strip()


def llmCompression(model, tokenizer, context):
    prompt = COMPRESSION_PROMPT.format(context=context)
    inputs = tokenizer(
        prompt, 
        add_special_tokens=False, 
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(**inputs, max_length=8000)
    compressed = tokenizer.decode(outputs[0]).split("<start_of_turn>model")[-1].strip()
    return compressed

COMPRESSION_PROMPT = """
<bos><start_of_turn>user
Extract and summarize the information and description of each products in the content.

content:
{context}<end_of_turn>
<start_of_turn>model
""".strip()


def llmCompression(pipe, context):
    prompt = COMPRESSION_PROMPT.format(context=context)
    output = pipe(
        prompt,
        max_lenth=8000,
        truncation=True,
        return_full_text=False
    )[0]["generated_text"]
    
    compressed = output.split("<end_of_turn>")[0].strip()
    return compressed
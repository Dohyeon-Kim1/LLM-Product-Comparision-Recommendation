import gradio as gr
from models import generation_pipe
from inference import inference

def main():

    pipe = generation_pipe("unsloth/gemma-2-2b-it") #모델 경로나 모델 id 필요!

    def answer(query):
        return inference(pipe,query)

    with gr.Blocks() as demo:
        question = gr.Textbox(lines=5, placeholder="비교 및 분석을 원하는 제품의 이름과 함께 질문해주세요 🤗",label = "User Question")
        output = gr.Textbox(label = "Answer")
        chat_btn = gr.Button("Let me know")
        chat_btn.click(
            fn=answer,
            inputs=question,
            outputs=output
            )

    demo.launch()

if __name__ == '__main__':
    main()


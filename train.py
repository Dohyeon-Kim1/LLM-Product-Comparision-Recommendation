import wandb
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq


USER_PROMPT = """
<start_of_turn>user
You are an expert product recommendation AI assistant trained to:
1. Compare products objectively based on features and specifications.
2. Provide personalized recommendations considering user needs.
3. Explain the pros and cons of each product clearly.
4. Support recommendations with factual information.

[System]
당신은 제품 추천과 비교를 전문으로 하는 AI 어시스턴트입니다. 주어진 컨텍스트를 기반으로 정확한 답변을 한국어로 생성해주세요.
마지막에는 항상 '추가적인 정보를 제공해주시면 보다 상세한 비교를 도와드리겠습니다! 🙂'라는 말을 붙여주세요.

[Instruction]
제공된 정보를 기반으로 질문에 답변해주세요. 만약 질문과 관련된 내용이 없더라도, 그 사실을 언급하지 말고 현재 보유한 지식을 활용하여 답변해주세요.

[Question]
{query}

[Context]
설명: {context}
<end_of_turn>
""".strip()


MODEL_RESPONSE = """
<start_of_turn>model
{response}
<end_of_turn>
""".strip()


def train(model, tokenizer, args):
    # wandb 초기화
    wandb.init(
        project = "Product Comparision & Recommendataion",
        config = {
            "model": "google/gemma-2-9b-it",
            "learning_rate": args.lr,
            "epochs": args.epoch,
            "batch_size": args.batch_size
        }
    )

    # 데이터 전처리*프롬프트 포맷 함수 정의
    def formatting_func(example, add_bos=True, add_eos=True):
        try:
            # 1. 필수 필드 검증(데이터셋의 필수 항목들이 모두 존재하는가)
            required_fields = ['system', 'instruction', 'query', 'context', 'response']
            for field in required_fields:
                if field not in example or not example[field]:
                    raise ValueError(f"Missing or empty required field: {field}")

            # 2. 텍스트 정리 함수(입력 텍스트 정리)
            def clean_text(text):
                return text.strip().replace('\n', ' ').replace('  ', ' ')  #strip(): 앞뒤 공백 제거
                                                                        #replace('\n', ' '): 줄바꿈 -> 공백 변환
                                                                        #replace('  ', ' '): 두 개 이상 연속 공백 하나로 줄임

            # 3. 시스템 프롬프트 포맷팅(ai의 역할과 작업 정의해줌)
            # 조건부 형식 사용해서 처리(bos: 시스템 프롬프트 시작 부분/eos: 모델 응답 끝 부분)
            # 4. 사용자 입력 포맷팅(사용자의 요청사항 구조화)
            user_query = USER_PROMPT.format(
                query=clean_text(example["query"]), 
                context=clean_text(example["context"])
            ) 
            user_query = f"<bos>{user_query}" if add_bos else user_query

            # 5. 모델 응답 포맷팅(ai의 응답 포맷 정의)
            model_response = MODEL_RESPONSE.format(response=example["response"])
            model_response = f"{model_response}<eos>" if add_eos else model_response

            # 6. 전체 프롬프트 조합(각 섹션을 하나의 프롬프트로 통합)
            prompt = f"{user_query}\n{model_response}"
            return prompt

        #에러처리
        except Exception as e:
            print(f"Error in formatting prompt: {str(e)}")
            return None  #에러 발생 시 none 반환

    # 데이터 전처리 함수
    def preprocess_function(example): #data 대신 example을 받음
        #단일 예제에 대해 formatting_func 적용
        text = formatting_func(example)

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=2048,
            padding=True,
            add_special_tokens=False
        )

        labels = encoded["input_ids"].copy()
        response_start = text.find("<start_of_turn>model")
        if response_start != -1:
            response_start_tokens = tokenizer(
                text[:response_start],
                add_special_tokens=False,
                truncation=True,
                max_length=2048
            )
            label_mask = [i < len(response_start_tokens["input_ids"]) for i in range(len(labels))]
            labels = [-100 if mask else lab for mask, lab in zip(label_mask, labels)]

        encoded["labels"] = labels
        return encoded

    # 데이터셋 불러오기
    dataset = load_dataset(args.dataset, split="train")

    #데이터셋 전처리 적용
    tokenized_dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        num_proc=4
    )

    # 데이터 콜레이터 설정
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    # 학습파라미터 설정
    training_args = TrainingArguments(
        output_dir= f"{args.output_dir}/checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size // args.gradient_accumulation_steps,
        learning_rate=args.lr,
        save_strategy="epoch",
        logging_dir="./logs",
        report_to="wandb",
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=True,
        fp16_opt_level="O2",
        gradient_checkpointing=True,
        lr_scheduler_type=args.lr_schedular,
    )

    # 트레이너 초기화
    trainer = Trainer(
        model=model, # 파인튜닝할 모델
        args=training_args, # 학습 파라미터
        train_dataset=tokenized_dataset, # 학습 데이터셋
        data_collator=data_collator, # 데이터 콜레이터
        tokenizer=tokenizer
    )

    # 모델 학습
    trainer.train()
    
    # 학습된 모델 저장
    model.save_pretrained_merged(
        f"{args.output_dir}/checkpoints/last",
        tokenizer,
        save_method="merged_16bit"
    )
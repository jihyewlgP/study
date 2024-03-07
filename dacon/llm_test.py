import pandas as pd
import numpy as np
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, AdamW
from tqdm import tqdm
import requests

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터 로드
data = pd.read_csv('D:/Dev/dacon/dacon/LLM/train.csv')

# SSL 보안 방화벽으로 인한 skt/kogpt2-base-v2는 불러오지 못함. 따라서 직접 내려받아서 가져와야함. => GPU가 아닌 cpu를 사용시 학습 정확도 매우 낮음 + 느림
# 토크나이저 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained('D:/Dev/dacon/dacon/LLM', bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

# 데이터 포맷팅 및 토크나이징
formatted_data = []
for _, row in tqdm(data.iterrows()):
    for q_col in ['질문_1', '질문_2']:
        for a_col in ['답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
            # 질문과 답변 쌍을 </s> token으로 연결
            input_text = row[q_col] + tokenizer.eos_token + row[a_col]
            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            formatted_data.append(input_ids)
print('Done.')

# 모델 로드
model = GPT2LMHeadModel.from_pretrained('D:/Dev/dacon/dacon/LLM')
model.to(device) # 모델을 GPU단으로 이동

# 모델 학습 하이퍼파라미터(Hyperparameter) 세팅
# 실제 필요에 따라 조정하세요.
CFG = {
    'LR' : 2e-5, # Learning Rate
    'EPOCHS' : 10, # 학습 Epoch
}

# 모델 학습 설정
optimizer = AdamW(model.parameters(), lr=CFG['LR'])
model.train()

# 모델 학습
for epoch in range(CFG['EPOCHS']):
    total_loss = 0
    progress_bar = tqdm(enumerate(formatted_data), total=len(formatted_data))
    for batch_idx, batch in progress_bar:
        # 데이터를 GPU단으로 이동
        batch = batch.to(device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # 진행률 표시줄에 평균 손실 업데이트
        progress_bar.set_description(f"Epoch {epoch+1} - Avg Loss: {total_loss / (batch_idx+1):.4f}")

    # 에폭의 평균 손실을 출력
    print(f"Epoch {epoch+1}/{CFG['EPOCHS']}, Average Loss: {total_loss / len(formatted_data)}")

# 모델 저장
model.save_pretrained("/DATA/dac/hansoldeco-kogpt2")
tokenizer.save_pretrained("/DATA/dac/hansoldeco-kogpt2")

# 저장된 Fine-tuned 모델과 토크나이저 불러오기
model_dir = "/DATA/dac/hansoldeco-kogpt2"
model = GPT2LMHeadModel.from_pretrained(model_dir)
model.to(device)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)

# Inference를 위한 test.csv 파일 로드
test = pd.read_csv('/DATA/dac/test.csv')

# test.csv의 '질문'에 대한 '답변'을 저장할 리스트
preds = []

# '질문' 컬럼의 각 질문에 대해 답변 생성
for test_question in tqdm(test['질문']):
    # 입력 텍스트를 토큰화하고 모델 입력 형태로 변환
    input_ids = tokenizer.encode(test_question + tokenizer.eos_token, return_tensors='pt')

    # 답변 생성
    output_sequences = model.generate(
        input_ids=input_ids.to(device),
        max_length=300,
        temperature=0.9,
        top_k=1,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1
    )

    # 생성된 텍스트(답변) 저장
    for generated_sequence in output_sequences:
        full_text = tokenizer.decode(generated_sequence, skip_special_tokens=False)
        # 질문과 답변의 사이를 나타내는 eos_token (</s>)를 찾아, 이후부터 출력
        answer_start = full_text.find(tokenizer.eos_token) + len(tokenizer.eos_token)
        answer_only = full_text[answer_start:].strip()
        answer_only = answer_only.replace('\n', ' ')
        preds.append(answer_only)
        
# Test 데이터셋의 모든 질의에 대한 답변으로부터 512 차원의 Embedding Vector 추출
# 평가를 위한 Embedding Vector 추출에 활용하는 모델은 'distiluse-base-multilingual-cased-v1' 이므로 반드시 확인해주세요.
from sentence_transformers import SentenceTransformer # SentenceTransformer Version 2.2.2

# Embedding Vector 추출에 활용할 모델(distiluse-base-multilingual-cased-v1) 불러오기
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# 생성한 모든 응답(답변)으로부터 Embedding Vector 추출
pred_embeddings = model.encode(preds)
pred_embeddings.shape

submit = pd.read_csv('/DATA/dac/sample_submission.csv')
# 제출 양식 파일(sample_submission.csv)을 활용하여 Embedding Vector로 변환한 결과를 삽입
submit.iloc[:,1:] = pred_embeddings
submit.head()

# 리더보드 제출을 위한 csv파일 생성
submit.to_csv('/DATA/dac/submit_version1.csv', index=False)
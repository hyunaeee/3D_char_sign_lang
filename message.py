import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, AdamW
import pandas as pd
import wandb

# wandb 초기화 및 config 설정
wandb.init(project="sign_language_translation", config={
    "learning_rate": 5e-5,
    "epochs": 3,
    "batch_size": 8,
    "max_source_length": 128,
    "max_target_length": 128,
})
config = wandb.config

# 1. 토크나이저와 모델 불러오기
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')

# wandb가 모델 가중치 변화를 추적하도록 watch 호출 (옵션)
wandb.watch(model)

# 2. 데이터셋 정의
class KOSignDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_source_length=config.max_source_length, max_target_length=config.max_target_length):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        source_text = self.df.loc[idx, 'korean_text']
        target_text = self.df.loc[idx, 'gloss_id']
        
        # 인코딩 (padding, truncation 포함)
        source_enc = self.tokenizer(source_text,
                                    max_length=self.max_source_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt")
        target_enc = self.tokenizer(target_text,
                                    max_length=self.max_target_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors="pt")
        
        # batch dimension 제거
        source_enc = {k: v.squeeze(0) for k, v in source_enc.items()}
        target_enc = {k: v.squeeze(0) for k, v in target_enc.items()}
        
        return source_enc, target_enc

# 3. 데이터 로더 생성
# 학습 및 검증 파일 경로 지정
train_csv = 'train.csv'
valid_csv = 'validation.csv'

# 각각의 데이터셋 생성
train_dataset = KOSignDataset(train_csv, tokenizer)
valid_dataset = KOSignDataset(valid_csv, tokenizer)

# DataLoader 생성: 학습 데이터는 shuffle=True, 검증 데이터는 shuffle=False
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

# 4. 학습 설정 (CUDA 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=config.learning_rate)
num_epochs = config.epochs  # wandb config에 설정한 epoch 수

# 5. 학습 루프 (매 에폭마다 검증 진행)
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    for batch_idx, (source, target) in enumerate(train_dataloader):
        input_ids = source['input_ids'].to(device)
        attention_mask = source['attention_mask'].to(device)
        labels = target['input_ids'].to(device)
        # pad token은 loss 계산 시 무시 (-100) 처리
        labels[labels == tokenizer.pad_token_id] = -100
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
            wandb.log({
            "step": epoch * len(train_dataloader) + batch_idx + 1,
            "train_loss": loss.item()
        })
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Training completed. Average Loss: {avg_train_loss:.4f}")
    
    # 검증 단계
    model.eval()
    total_valid_loss = 0.0
    with torch.no_grad():
        for batch_idx, (source, target) in enumerate(valid_dataloader):
            input_ids = source['input_ids'].to(device)
            attention_mask = source['attention_mask'].to(device)
            labels = target['input_ids'].to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_valid_loss += loss.item()
    
    avg_valid_loss = total_valid_loss / len(valid_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_valid_loss:.4f}")
    
    # wandb로 에폭별 loss 로깅
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "validation_loss": avg_valid_loss
    })

# 6. 모델 및 토크나이저 저장 (옵션)
model.save_pretrained("kobart_finetuned_kosign")
tokenizer.save_pretrained("kobart_finetuned_kosign")

# 7. 제약 조건을 적용한 추론 (constrained decoding)

# (1) 데이터셋에서 gloss_id 컬럼의 형태소들을 추출하는 함수 정의
def extract_morph_tokens(dataset):
    """
    dataset 객체의 df['gloss_id']에 있는 문자열을 공백 기준으로 분리하여
    고유한 형태소(token) 집합을 반환합니다.
    """
    morph_tokens = set()
    for gloss in dataset.df["gloss_id"]:
        # 예: "오늘1 밤1 시:9시 춥다1" 인 경우 공백을 기준으로 분리
        tokens = gloss.split()
        morph_tokens.update(tokens)
    return list(morph_tokens)

# (2) train과 validation 데이터셋에서 형태소 추출 후 합집합 생성
train_morph_tokens = set(extract_morph_tokens(train_dataset))
valid_morph_tokens = set(extract_morph_tokens(valid_dataset))
allowed_gloss_tokens = list(train_morph_tokens.union(valid_morph_tokens))

# # (3) allowed_gloss_tokens에 포함된 각 형태소가 토크나이저에 의해 하나의 토큰으로 인코딩되는지 확인하고, 
# #     allowed_token_ids 집합에 추가합니다.
# allowed_token_ids = set()
# for token in allowed_gloss_tokens:
#     token_ids = tokenizer(token, add_special_tokens=False)['input_ids']
#     if len(token_ids) == 1:  # 하나의 토큰이어야 함
#         allowed_token_ids.add(token_ids[0])
#     else:
#         # 만약 한 단어가 여러 토큰으로 분리된다면, warn 메시지를 출력합니다.
#         print(f"Warning: '{token}' is tokenized into multiple tokens: {token_ids}")

# # 생성 종료를 위해 EOS 토큰도 반드시 허용합니다.
# allowed_token_ids.add(tokenizer.eos_token_id)
# allowed_token_ids = list(allowed_token_ids)

# 먼저, allowed_gloss_tokens에 있는 형태소들을 토큰 시퀀스로 변환하여 allowed_sequences 사전을 구성합니다.
# allowed_sequences는 각 형태소 문자열을 키로, 그에 대응하는 토큰 id 리스트를 값으로 가집니다.
allowed_sequences = {}
for gloss in allowed_gloss_tokens:
    token_ids = tokenizer(gloss, add_special_tokens=False)['input_ids']
    allowed_sequences[gloss] = token_ids

# prefix_allowed_tokens_fn 수정:
def prefix_allowed_tokens_fn(batch_id, input_ids):
    """
    현재까지 생성된 input_ids를 기반으로, allowed_sequences에 있는 형태소 중
    해당 토큰 시퀀스의 접두사와 일치하는 경우 다음에 허용할 토큰을 반환합니다.
    만약 이미 전체 allowed 시퀀스가 생성되었다면, EOS 토큰만 허용합니다.
    """
    # 현재까지 생성된 토큰 시퀀스를 리스트로 변환합니다.
    generated = input_ids.tolist()
    allowed_next_tokens = set()

    # 각 allowed 시퀀스에 대해 현재까지 생성된 토큰과의 일치 여부를 확인합니다.
    for seq in allowed_sequences.values():
        # 생성된 토큰이 해당 allowed 시퀀스의 시작 부분과 일치하는 경우
        if generated == seq[:len(generated)]:
            # 만약 전체 시퀀스가 이미 생성되었다면, 종료를 위해 EOS 토큰을 허용합니다.
            if len(generated) == len(seq):
                allowed_next_tokens.add(tokenizer.eos_token_id)
            else:
                # 아직 전체 시퀀스가 완성되지 않았다면, 다음에 나와야 할 토큰을 허용합니다.
                allowed_next_tokens.add(seq[len(generated)])
    
    # 초기 상태(아무 토큰도 생성되지 않은 경우)라면 각 allowed 시퀀스의 첫 토큰들을 모두 허용합니다.
    if len(generated) == 0:
        for seq in allowed_sequences.values():
            allowed_next_tokens.add(seq[0])
    
    # 만약 어떤 allowed 시퀀스와도 매칭되지 않는 경우 (예상치 못한 상황)엔 빈 리스트 대신 EOS만 허용해 추론을 종료합니다.
    if not allowed_next_tokens:
        allowed_next_tokens.add(tokenizer.eos_token_id)
    
    return list(allowed_next_tokens)

# 이후 generate() 호출 시 prefix_allowed_tokens_fn을 전달하면,
# 모델은 각 단계마다 allowed_sequences에 해당하는 토큰만 출력하도록 제한됩니다.
model.eval()
sample_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
with torch.no_grad():
    for batch in sample_dataloader:
        input_ids = batch[0]['input_ids'].to(device)
        attention_mask = batch[0]['attention_mask'].to(device)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            print(decoded)

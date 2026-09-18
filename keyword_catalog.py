"""키워드 추상화 단계(tier) 분류 체계.

가장 기초적인 수학/연산 지식부터 가장 상위의 패러다임/카테고리까지
7단계(0~6)로 키워드를 분류한다. 색상은 프론트엔드(styles.css)에서
tier id 기준으로 매핑하므로, 여기서는 색을 다루지 않는다.

논문 데이터는 arXiv에서 실시간으로 가져오므로(arxiv_service.py), 사람이 논문마다
키워드를 미리 붙여둘 수 없다. 대신 이 파일의 extract_keywords() 가 논문 제목/초록
텍스트를 KEYWORD_CATALOG 및 KEYWORD_ALIASES 와 대조해 언급된 키워드를 찾아낸다.
"""

import re

TIERS = [
    {
        "id": 0,
        "name": "수학 기초",
        "description": "다른 모든 키워드의 전제가 되는 수학/통계 분야 자체.",
    },
    {
        "id": 1,
        "name": "연산 / 메커니즘",
        "description": "구체적인 계산식이나 알고리즘 수준의 핵심 동작.",
    },
    {
        "id": 2,
        "name": "레이어 / 학습 기법",
        "description": "신경망 레이어 구성 요소나 학습 과정에서 쓰이는 기법.",
    },
    {
        "id": 3,
        "name": "최적화 기법",
        "description": "파라미터 업데이트 방식이나 학습 목적함수를 조정하는 기법.",
    },
    {
        "id": 4,
        "name": "구성 블록 / 소규모 아키텍처",
        "description": "여러 레이어가 묶여 하나의 기능 단위를 이루는 블록.",
    },
    {
        "id": 5,
        "name": "대규모 아키텍처",
        "description": "여러 블록이 결합되어 완성된 모델 전체 구조.",
    },
    {
        "id": 6,
        "name": "상위 패러다임 / 카테고리",
        "description": "특정 아키텍처를 넘어서는 학습 방법론이나 시스템 구성, AI 철학.",
    },
]

TIER_BY_ID = {tier["id"]: tier for tier in TIERS}

# 키워드 이름 -> tier id
KEYWORD_CATALOG = {
    # tier 0: 수학 기초
    "선형대수": 0,
    "확률/통계": 0,
    "미분": 0,
    "베이즈 통계": 0,
    "최적화 이론": 0,
    "그래프 이론": 0,
    "스펙트럴 그래프 이론": 0,
    "게임이론(min-max)": 0,
    "마르코프 결정과정(MDP)": 0,
    "동적계획법": 0,
    "확률과정(마르코프체인)": 0,
    "SDE 기초": 0,
    "트리 탐색 이론": 0,
    # tier 1: 연산 / 메커니즘
    "Softmax": 1,
    "Sigmoid": 1,
    "ReLU": 1,
    "GELU": 1,
    "SwiGLU": 1,
    "Cross-Entropy": 1,
    "코사인 유사도": 1,
    "KL Divergence": 1,
    "변분추론": 1,
    "Latent Variable": 1,
    "역전파": 1,
    "Self-Attention": 1,
    "Attention Mechanism": 1,
    "Flash Attention": 1,
    "Denoising": 1,
    "Q-Learning": 1,
    "Policy Gradient": 1,
    "Monte Carlo Tree Search": 1,
    "Contrastive Loss": 1,
    "Word Embedding": 1,
    "Message Passing": 1,
    # tier 2: 레이어 / 학습 기법
    "Multi-Head Attention": 2,
    "Positional Encoding": 2,
    "Rotary Position Embedding": 2,
    "Layer Normalization": 2,
    "Batch Normalization": 2,
    "RMSNorm": 2,
    "Dropout": 2,
    "Masked Language Model": 2,
    "Fine-tuning": 2,
    "Residual Connection": 2,
    "Patch Embedding": 2,
    "Experience Replay": 2,
    "Graph Convolution": 2,
    "Image-Text Embedding": 2,
    "Data Augmentation": 2,
    "Autoregressive LM": 2,
    # tier 3: 최적화 기법
    "SGD": 3,
    "Momentum": 3,
    "Adam": 3,
    "AdamW": 3,
    "Adagrad": 3,
    "RMSProp": 3,
    "Weight Decay": 3,
    "Learning Rate Warmup": 3,
    "Gradient Clipping": 3,
    "Adversarial Training": 3,
    "Clipped Objective": 3,
    "PPO": 3,
    # tier 4: 구성 블록 / 소규모 아키텍처
    "CNN": 4,
    "RNN": 4,
    "LSTM": 4,
    "MLP": 4,
    "GNN": 4,
    "Transformer Encoder": 4,
    "Generator": 4,
    "Discriminator": 4,
    "Encoder-Decoder": 4,
    "Actor-Critic": 4,
    "Policy Network": 4,
    "Value Network": 4,
    "Reward Model": 4,
    "Skip-gram": 4,
    "CBOW": 4,
    # tier 5: 대규모 아키텍처
    "Transformer": 5,
    "ResNet": 5,
    "Vision Transformer": 5,
    "GPT": 5,
    "BERT": 5,
    "LLaMA": 5,
    "GAN": 5,
    "Variational Autoencoder": 5,
    "U-Net": 5,
    "Deep Q-Network": 5,
    "GCN": 5,
    "CLIP": 5,
    # tier 6: 상위 패러다임 / 카테고리
    "In-context Learning": 6,
    "Scaling Law": 6,
    "Pre-trained Language Model": 6,
    "Diffusion Model": 6,
    "Contrastive Learning": 6,
    "Zero-shot Transfer": 6,
    "Self-Supervised Learning": 6,
    "RLHF": 6,
    "MoE": 6,
    "AI Agent": 6,
    "MLOps": 6,
}


def resolve_keyword(name):
    tier_id = KEYWORD_CATALOG.get(name)
    tier = TIER_BY_ID.get(tier_id)
    return {
        "name": name,
        "tier": tier_id,
        "tier_name": tier["name"] if tier else None,
    }


# arXiv 초록은 거의 항상 영어라서, 한글로만 표기된 수학 기초 키워드는 이 영어
# 동의어들로 텍스트를 스캔해야 실제로 매칭된다. 등록되지 않은 키워드는 자기 이름
# 자체를 패턴으로 사용한다 (영문 기술/아키텍처 키워드가 대부분 여기 해당).
KEYWORD_ALIASES = {
    "선형대수": ["linear algebra"],
    "확률/통계": ["probability", "statistics", "probabilistic"],
    "미분": ["calculus", "differentiation"],
    "베이즈 통계": ["bayesian"],
    "최적화 이론": ["optimization theory", "convex optimization"],
    "그래프 이론": ["graph theory"],
    "스펙트럴 그래프 이론": ["spectral graph"],
    "게임이론(min-max)": ["game theory", "minimax", "min-max"],
    "마르코프 결정과정(MDP)": ["markov decision process", "MDP"],
    "동적계획법": ["dynamic programming"],
    "확률과정(마르코프체인)": ["markov chain"],
    "SDE 기초": ["stochastic differential equation", "SDE"],
    "트리 탐색 이론": ["tree search"],
    "역전파": ["backpropagation", "back-propagation"],
    "코사인 유사도": ["cosine similarity"],
    "변분추론": ["variational inference"],
    "Vision Transformer": ["Vision Transformer", "ViT"],
    "Variational Autoencoder": ["Variational Autoencoder", "VAE"],
    "Deep Q-Network": ["Deep Q-Network", "DQN"],
    "GCN": ["GCN", "Graph Convolutional Network"],
    "Monte Carlo Tree Search": ["Monte Carlo Tree Search", "MCTS"],
    "Cross-Entropy": ["Cross-Entropy", "Cross Entropy"],
    "Pre-trained Language Model": ["Pre-trained Language Model", "Pretrained Language Model"],
    "Self-Supervised Learning": ["Self-Supervised Learning", "Self Supervised"],
    "Flash Attention": ["Flash Attention", "FlashAttention", "flash-attention"],
    "Layer Normalization": ["Layer Normalization", "LayerNorm", "Layer Norm", "LN"],
    "Batch Normalization": ["Batch Normalization", "BatchNorm", "Batch Norm", "BN"],
    "Rotary Position Embedding": ["Rotary Position Embedding", "RoPE"],
    "SGD": ["SGD", "Stochastic Gradient Descent"],
    "Learning Rate Warmup": ["Learning Rate Warmup", "LR Warmup", "Learning Rate Scheduling", "Learning Rate Schedule"],
    "GNN": ["GNN", "Graph Neural Network"],
    "RNN": ["RNN", "Recurrent Neural Network"],
    "LSTM": ["LSTM", "Long Short-Term Memory"],
    "MLP": ["MLP", "Multi-Layer Perceptron", "Multilayer Perceptron"],
    "MoE": ["MoE", "Mixture of Experts", "Mixture-of-Experts"],
    "AI Agent": ["AI agent", "autonomous agent", "LLM agent", "agentic"],
}


def extract_keywords(text, limit=8):
    """텍스트(제목+초록)에서 언급된 카탈로그 키워드를 tier 오름차순으로 찾아낸다."""
    haystack = text.lower()
    found = []
    for name in KEYWORD_CATALOG:
        patterns = KEYWORD_ALIASES.get(name, [name])
        if any(_mentions(haystack, pattern) for pattern in patterns):
            found.append(resolve_keyword(name))
    found.sort(key=lambda kw: kw["tier"])
    return found[:limit]


def _mentions(haystack, pattern):
    escaped = re.escape(pattern.lower())
    return re.search(rf"\b{escaped}\b", haystack) is not None

"""유명 AI/ML 논문들의 예시 커리큘럼 — 실제로 Gemini/arXiv 파이프라인을 돌려서
얻은 결과를 그대로 박아둔 정적 데이터.

앱이 시작될 때 DB에 예시가 하나도 없으면 curriculum_store.seed_example_curricula_if_missing()이
이 데이터를 그대로 심는다. 배포 환경(Fly.io)은 data/app.db용 영구 볼륨이 없어서
재배포마다 DB가 초기화되는데, 그때마다 Gemini/arXiv를 다시 호출해 예시를
재생성하면(예: Dockerfile 빌드 단계에 넣는 방식) 커밋할 때마다 quota를 쓰게 되고
API 키를 빌드 레이어에 노출해야 하는 문제가 있었다 — 그래서 "한 번 실제로
생성한 결과를 커밋해두고, 매 시작 시 그대로 집어넣기만" 하는 방식을 택했다.
새 예시를 추가/갱신하려면 seed_examples.py를 다시 돌리면(생성 + 이 파일 재작성을
한 번에 함) 된다. 이 파일은 손으로 고치지 말 것 — seed_examples.py가 매번
새로 써서 덮어쓴다.
"""

EXAMPLE_CURRICULA = [{'domain': 'ai_ml',
  'edges': [{'from': 'n1', 'to': 'n4'},
            {'from': 'n2', 'to': 'n3'},
            {'from': 'n3', 'to': 'n5'},
            {'from': 'n4', 'to': 'n5'}],
  'nodes': [{'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': '확률/통계', 'tier': 0, 'tier_name': '수학 기초'}],
             'description': '확률 및 통계 기초에서는 데이터 분포를 표현하고 추정하는 기본적인 수학적 틀을 다룬다. 이 노드는 D2L 교재에서 다루는 '
                            '데이터 생성 분포와 확률 변수 개념의 바탕이 되며, 이후 제너레이터가 암묵적으로 정의하는 확률 분포 pg와 실제 데이터 분포 '
                            'pdata를 비교하는 이론적 기반을 제공한다. 특히 쿨백-라이블러 다이버전스(Kullback-Leibler '
                            'Divergence)와 젠슨-섀넌 다이버전스를 이해하기 위한 필수적인 선수 조건이다. 학습자는 확률 밀도 함수와 기대값 '
                            '개념을 통해 GAN의 목적 함수를 수식적으로 분석할 수 있게 된다.',
             'id': 'n1',
             'is_target': False,
             'layer': 0,
             'learning_points': ['연속 확률 변수와 확률 밀도 함수의 기본 개념을 설명할 수 있다.',
                                 '기대값 연산과 확률 분포 간의 차이를 측정하는 직관을 갖춘다.'],
             'quiz': None,
             'title': '확률 및 통계 기초'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': '미분', 'tier': 0, 'tier_name': '수학 기초'},
                          {'name': 'SGD', 'tier': 3, 'tier_name': '최적화 기법'}],
             'description': '미분과 최적화 기초는 모델의 파라미터를 조정하여 목적 함수를 최소화하거나 최대화하는 방법을 다룬다. 이 논문 PDF 원문에서 '
                            '언급하듯, 제너레이터와 디스크리미네이터는 역전파(backpropagation)를 통해 전체 시스템이 학습되며, 이를 위해 '
                            '경사하강법과 서브디리바티브(subderivatives) 개념이 활용된다. 이 노드는 미니맥스 게임의 최적점을 찾기 위한 '
                            '최적화 과정을 이해하는 데 직접적인 도구를 제공한다. 학습자는 목적 함수의 미분을 통해 그래디언트의 흐름을 파악할 수 있다.',
             'id': 'n2',
             'is_target': False,
             'layer': 0,
             'learning_points': ['함수의 미분 값을 바탕으로 경사하강법의 업데이트 단계를 수행할 수 있다.',
                                 '목적 함수의 변화에 따른 파라미터 최적화 방향을 도출할 수 있다.'],
             'quiz': None,
             'title': '미분과 최적화 기초'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'MLP', 'tier': 4, 'tier_name': '구성 블록 / 소규모 아키텍처'},
                          {'name': 'Sigmoid', 'tier': 1, 'tier_name': '연산 / 메커니즘'}],
             'description': '확률적 신경망과 MLP(Multilayer Perceptron)는 다층 퍼셉트론을 활용하여 복잡한 비선형 함수를 근사하는 '
                            '방법을 다룬다. 이 논문 PDF 원문에서는 제너레이터와 디스크리미네이터가 다층 퍼셉트론으로 정의될 때 역전파를 통해 전체 '
                            '시스템이 훈련된다고 설명한다. D2L 교재에서도 디스크리미네이터를 구성할 때 3층의 MLP를 사용하여 실제 데이터와 생성된 '
                            '데이터를 구분하는 이진 로지스틱 회귀를 수행한다고 언급한다. 이 노드는 GAN의 양대 신경망 아키텍처를 구현하기 위한 '
                            '직접적인 토대가 된다.',
             'id': 'n3',
             'is_target': False,
             'layer': 1,
             'learning_points': ['다층 퍼셉트론의 구조와 순방향 및 역전파 연산 과정을 설명할 수 있다.',
                                 '비선형 활성화 함수가 신경망의 표현력에 미치는 영향을 이해한다.'],
             'quiz': None,
             'title': '확률적 신경망과 MLP'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': '게임이론(min-max)', 'tier': 0, 'tier_name': '수학 기초'},
                          {'name': 'Cross-Entropy', 'tier': 1, 'tier_name': '연산 / 메커니즘'}],
             'description': '게임 이론과 미니맥스 최적화는 두 플레이어가 서로 상반된 목적을 가지고 경쟁하는 프레임워크를 다룬다. 이 논문 PDF '
                            '원문에서는 제너레이터 G가 디스크리미네이터 D의 오답 확률을 최대화하고, D는 진짜와 가짜를 판별하는 미니맥스 2인 게임으로 '
                            'GAN 프레임워크를 정의한다. 이 노드는 가상의 훈련 기준 C(G)의 전역 최소값이 pdata = pg일 때 달성되며, 이때 '
                            '젠슨-섀넌 다이버전스와 연관됨을 이해하는 데 핵심적인 역할을 한다.',
             'id': 'n4',
             'is_target': False,
             'layer': 1,
             'learning_points': ['미니맥스 게임의 수학적 정의와 균형점의 의미를 설명할 수 있다.',
                                 '두 플레이어의 상호작용이 최적화 과정에서 어떻게 균형을 이루는지 분석할 수 있다.'],
             'quiz': None,
             'title': '게임 이론과 미니맥스 최적화'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'Generator', 'tier': 4, 'tier_name': '구성 블록 / 소규모 아키텍처'},
                          {'name': 'Discriminator', 'tier': 4, 'tier_name': '구성 블록 / 소규모 아키텍처'},
                          {'name': 'Adam', 'tier': 3, 'tier_name': '최적화 기법'},
                          {'name': 'Dropout', 'tier': 2, 'tier_name': '레이어 / 학습 기법'}],
             'description': 'Generative Adversarial Networks(GAN)는 적대적 과정을 통해 생성 모델을 추정하는 새로운 '
                            '프레임워크이다. 이 논문 PDF 원문과 D2L 교재에 따르면, 데이터 분포를 포착하는 제너레이터 G와 샘플이 훈련 데이터에서 '
                            '왔을 확률을 추정하는 디스크리미네이터 D를 동시에 훈련시킨다. 초기 학습에서 G가 미숙할 때 log(1 - D(G(z)))가 '
                            '포화되는 문제를 해결하기 위해 log D(G(z))를 최대화하도록 G를 훈련하는 기법을 사용하며, 마르코프 체인 없이 '
                            '역전파만으로 훈련이 가능하다. 학습자는 이 최종 노드를 통해 생성 모델의 이론적 최적점과 실제 훈련 알고리즘을 완벽히 '
                            '이해하게 된다.',
             'id': 'n5',
             'is_target': True,
             'layer': 2,
             'learning_points': ['제너레이터와 디스크리미네이터의 상호 훈련 알고리즘을 코드로 구현할 수 있다.',
                                 '가상의 훈련 기준과 젠슨-섀넌 다이버전스의 관계를 증명 과정을 통해 설명할 수 있다.',
                                 '헬베치카 시나리오나 포화 현상 등 GAN 학습 시 발생하는 한계와 해결책을 파악할 수 있다.'],
             'quiz': None,
             'title': 'Generative Adversarial Networks'}],
  'pdf_url': 'https://arxiv.org/pdf/1406.2661v1',
  'target_description': 'We propose a new framework for estimating generative models via an '
                        'adversarial process, in which we simultaneously train two models: a '
                        'generative model G that captures the data distribution, and a '
                        'discriminative model D that estimates the probability that a sample came '
                        'from the training data rather than G. The training procedure for G is to '
                        'maximize the probability of D making a mistake. This framework '
                        'corresponds to a minimax two-player game. In the space of arbitrary '
                        'functions G and D, a unique solution exists, with G recovering the '
                        'training data distribution and D equal to 1/2 everywhere. In the case '
                        'where G and D are defined by multilayer perceptrons, the entire system '
                        'can be trained with backpropagation. There is no need for any Markov '
                        'chains or unrolled approximate inference networks during either training '
                        'or generation of samples. Experiments demonstrate the potential of the '
                        'framework through qualitative and quantitative evaluation of the '
                        'generated samples.',
  'target_label': 'Generative Adversarial Networks',
  'target_type': 'paper',
  'used_rag': True},
 {'domain': 'ai_ml',
  'edges': [{'from': 'n1', 'to': 'n2'},
            {'from': 'n1', 'to': 'n3'},
            {'from': 'n2', 'to': 'n4'},
            {'from': 'n3', 'to': 'n4'},
            {'from': 'n4', 'to': 'n5'},
            {'from': 'n5', 'to': 'n6'},
            {'from': 'n6', 'to': 'n7'},
            {'from': 'n7', 'to': 'n8'}],
  'nodes': [{'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': '미분', 'tier': 0, 'tier_name': '수학 기초'},
                          {'name': '선형대수', 'tier': 0, 'tier_name': '수학 기초'}],
             'description': '이 노드에서는 딥러닝 모델 학습의 근간이 되는 수학적 기초와 미분 개념을 다룬다. [D2L 교재]에 따르면 선형대수, 미분, '
                            '확률 등 초보적인 수치 연산을 통해 데이터 조작과 수치 계산 능력을 배양해야 한다. 이를 통해 향후 신경망의 오차 역전파와 '
                            '가중치 갱신 메커니즘을 이해할 수 있는 수학적 토대를 단단하게 다진다.',
             'id': 'n1',
             'is_target': False,
             'layer': 0,
             'learning_points': ['기본적인 선형대수 및 미분 연산을 코드로 구현할 수 있다.',
                                 '미분 개념을 활용하여 함수의 변화율을 계산할 수 있다.'],
             'quiz': None,
             'title': '기초 수학 및 미분'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'ReLU', 'tier': 1, 'tier_name': '연산 / 메커니즘'},
                          {'name': 'Softmax', 'tier': 1, 'tier_name': '연산 / 메커니즘'}],
             'description': '이 노드에서는 신경망의 비선형성을 부여하는 활성화 함수인 ReLU와 Softmax를 학습한다. [이 논문 PDF 원문]에서 '
                            'ResNet 내부의 빌딩 블록은 ReLU 등의 활성화 함수를 통해 비선형 매핑을 수행하며, 편향이 생략되거나 두 번째 '
                            '비선형성이 덧셈 연산 후에 배치된다고 설명한다. 이 개념은 추후 레이어들이 복잡한 잔차 함수를 학습할 수 있게 하는 필수적 '
                            '메커니즘을 이해하게 도와준다.',
             'id': 'n2',
             'is_target': False,
             'layer': 1,
             'learning_points': ['ReLU 함수와 Softmax 함수의 수식적 차이와 역할을 설명할 수 있다.',
                                 '비선형 활성화 함수가 신경망 표현력에 미치는 영향을 이해한다.'],
             'quiz': None,
             'title': '활성화 함수 (ReLU와 Softmax)'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'multilayer perceptrons', 'tier': 2, 'tier_name': '레이어 / 학습 기법'},
                          {'name': '과적합', 'tier': 2, 'tier_name': '레이어 / 학습 기법'}],
             'description': '이 노드에서는 회귀, 분류, 다층 퍼셉트론, 그리고 과적합 및 정규화 같은 가장 기본적인 딥러닝 기법들을 학습한다. [D2L '
                            '교재]에서는 이 단계가 컴퓨터 시퀀셜 데이터나 비전 시스템을 다루기 전 거쳐야 할 가장 기본적인 딥러닝 테크닉들을 다룬다고 '
                            '언급한다. 이를 충실히 익혀야 추후 심층 신경망에서 발생하는 현상들을 정확히 분석할 수 있다.',
             'id': 'n3',
             'is_target': False,
             'layer': 1,
             'learning_points': ['다층 퍼셉트론의 구조를 파악하고 기본 분류 문제를 풀 수 있다.',
                                 '과적합과 정규화의 개념을 정의하고 대응 방안을 제시할 수 있다.'],
             'quiz': None,
             'title': '딥러닝 기본 학습과 과적합'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'Degradation', 'tier': 3, 'tier_name': '최적화 기법'}],
             'description': '이 노드에서는 망의 깊이가 깊어질수록 오히려 정확도가 떨어지고 훈련 오차가 높아지는 성능 저하 문제를 다룬다. [이 논문 '
                            'PDF 원문]에 따르면, 이러한 현상은 과적합으로 인한 것이 아니며 현재 사용되는 솔버들이 최적의 솔루션을 찾지 못해 '
                            '발생한다. 이 문제를 직시함으로써 왜 기존의 단순한 망 구조를 넘어 새로운 학습 프레임워크가 필요한지 명확한 동기를 부여받게 '
                            '된다.',
             'id': 'n4',
             'is_target': False,
             'layer': 2,
             'learning_points': ['신경망 깊이가 깊어질 때 훈련 오차가 증가하는 열화 문제를 설명할 수 있다.',
                                 '성능 저하가 과적합에 의한 것이 아님을 실험적 증거를 통해 이해한다.'],
             'quiz': None,
             'title': '심층 신경망의 성능 저하 문제 (Degradation Problem)'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'Residual Learning', 'tier': 3, 'tier_name': '최적화 기법'}],
             'description': '이 노드에서는 원하는 매핑 H(x) 대신 잔차 매핑 F(x) := H(x) - x를 학습하도록 재구성하는 핵심 접근법을 '
                            '배운다. [이 논문 PDF 원문]에서는 층들이 원래의 비참조 함수 대신 잔차 함수를 학습하게 하면 최적화가 훨씬 쉬워진다고 '
                            '가정한다. 이를 통해 항등 매핑이 최적일 때 퍼섭테이션을 쉽게 찾을 수 있도록 문제를 사전 조건화하는 방법을 익히게 된다.',
             'id': 'n5',
             'is_target': False,
             'layer': 3,
             'learning_points': ['원래의 매핑 H(x)를 잔차 함수 F(x) + x 형태로 재구성하는 수식을 이해한다.',
                                 '잔차 매핑 학습이 기존 unreferenced 매핑 학습보다 최적화하기 쉬운 이유를 설명할 수 있다.'],
             'quiz': None,
             'title': '잔차 학습 프레임워크 (Residual Learning)'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'Shortcut Connections',
                           'tier': 4,
                           'tier_name': '구성 블록 / 소규모 아키텍처'}],
             'description': '이 노드에서는 입력 x를 출력에 직접 더해주는 숏컷 커넥션과 파라미터가 없는 항등 매핑 구조를 학습한다. [이 논문 PDF '
                            '원문]에서는 y = F(x, {Wi}) + x로 정의되는 빌딩 블록을 소개하며, 이 파라미터 없는 숏컷이 복잡도를 늘리지 '
                            '않으면서도 정보가 손실 없이 전달되도록 돕는다고 설명한다. 이를 통해 앞서 배운 잔차 학습을 실제 레이어 구조로 구현하는 '
                            '방법을 완성한다.',
             'id': 'n6',
             'is_target': False,
             'layer': 4,
             'learning_points': ['파라미터가 없는 항등 숏컷 커넥션의 구조와 수식을 작성할 수 있다.',
                                 '차원이 변경될 때 선형 프로젝션 Ws를 사용하는 옵션을 비교할 수 있다.'],
             'quiz': None,
             'title': '숏컷 커넥션과 항등 매핑 (Shortcut Connections)'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'ResNet', 'tier': 5, 'tier_name': '대규모 아키텍처'}],
             'description': '이 노드에서는 1x1, 3x3, 1x1 컨볼루션을 조합하여 연산량을 줄이면서 깊은 층을 쌓는 보틀넥 빌딩 블록을 학습한다. '
                            '[이 논문 PDF 원문]에서는 50층, 101층, 152층의 ResNet 모델을 구축할 때 연산 효율성을 유지하기 위해 '
                            '보틀넥 구조가 필수적이며, 항등 숏컷과 결합하여 모델 크기와 연산량을 획기적으로 낮춘다고 설명한다. 이 아키텍처는 최종 '
                            '목표인 ResNet 논문의 성능 검증 단계로 직접 연결된다.',
             'id': 'n7',
             'is_target': False,
             'layer': 5,
             'learning_points': ['1x1 및 3x3 컨볼루션을 활용한 보틀넥 블록의 구조를 그릴 수 있다.',
                                 '대규모 네트워크에서 보틀넥 디자인이 연산 시간과 파라미터 수에 미치는 영향을 분석할 수 있다.'],
             'quiz': None,
             'title': '보틀넥 아키텍처와 ResNet'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'ResNet', 'tier': 5, 'tier_name': '대규모 아키텍처'}],
             'description': '이 노드는 학습의 최종 목표로서, 이미지 인식에서의 딥 잔차 학습 논문의 전체적인 기여와 성과를 통합적으로 이해한다. [이 '
                            '논문 PDF 원문]에 따르면 152층의 ResNet은 VGG보다 깊으면서도 더 낮은 복잡도를 가지며, ILSVRC 2015 '
                            '및 COCO 2015 대회에서 1위를 차지하는 등 컴퓨터 비전 전반에 혁신적인 성과를 거두었다. 앞서 배운 수학, 활성화 '
                            '함수, 잔차 학습, 보틀넥 아키텍처 등의 모든 지식이 이 최종 목표 안에서 하나로 종합된다.',
             'id': 'n8',
             'is_target': True,
             'layer': 6,
             'learning_points': ['ILSVRC 2015에서 1위를 차지한 ResNet 모델의 핵심 기여와 실험 결과를 요약할 수 있다.',
                                 '극도로 깊은 표현력이 COCO 및 ImageNet 등의 시각 인식 과제에 미친 성능 향상을 분석할 수 있다.'],
             'quiz': None,
             'title': 'Deep Residual Learning for Image Recognition'}],
  'pdf_url': 'https://arxiv.org/pdf/1512.03385v1',
  'target_description': 'Deeper neural networks are more difficult to train. We present a residual '
                        'learning framework to ease the training of networks that are '
                        'substantially deeper than those used previously. We explicitly '
                        'reformulate the layers as learning residual functions with reference to '
                        'the layer inputs, instead of learning unreferenced functions. We provide '
                        'comprehensive empirical evidence showing that these residual networks are '
                        'easier to optimize, and can gain accuracy from considerably increased '
                        'depth. On the ImageNet dataset we evaluate residual nets with a depth of '
                        'up to 152 layers---8x deeper than VGG nets but still having lower '
                        'complexity. An ensemble of these residual nets achieves 3.57% error on '
                        'the ImageNet test set. This result won the 1st place on the ILSVRC 2015 '
                        'classification task. We also present analysis on CIFAR-10 with 100 and '
                        '1000 layers. The depth of representations is of central importance for '
                        'many visual recognition tasks. Solely due to our extremely deep '
                        'representations, we obtain a 28% relative improvement on the COCO object '
                        'detection dataset. Deep residual nets are foundations of our submissions '
                        'to ILSVRC & COCO 2015 competitions, where we also won the 1st places on '
                        'the tasks of ImageNet detection, ImageNet localization, COCO detection, '
                        'and COCO segmentation.',
  'target_label': 'Deep Residual Learning for Image Recognition',
  'target_type': 'paper',
  'used_rag': True},
 {'domain': 'ai_ml',
  'edges': [{'from': 'n1', 'to': 'n3'},
            {'from': 'n2', 'to': 'n3'},
            {'from': 'n3', 'to': 'n4'},
            {'from': 'n4', 'to': 'n5'},
            {'from': 'n4', 'to': 'n6'},
            {'from': 'n5', 'to': 'n6'},
            {'from': 'n6', 'to': 'n8'},
            {'from': 'n7', 'to': 'n8'}],
  'nodes': [{'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'RNN', 'tier': 4, 'tier_name': '구성 블록 / 소규모 아키텍처'},
                          {'name': 'LSTM', 'tier': 4, 'tier_name': '구성 블록 / 소규모 아키텍처'}],
             'description': '이 노드에서는 [이 논문 PDF 원문]에서 다루는 Recurrent neural networks와 long '
                            'short-term memory(LSTM)의 구조적 한계를 학습한다. 순환 모델은 입력 및 출력 시퀀스의 심볼 위치에 따라 '
                            '연산을 고정하므로, 이전 은닉 상태와 현재 입력을 순차적으로 처리해야 하는 내재적 순차성이 발생한다. 이러한 '
                            'inherently sequential nature는 훈련 예제 내에서의 병렬화를 원천적으로 차단하며, 특히 더 긴 시퀀스 '
                            '길이에서 메모리 제약으로 인해 큰 병목을 유발한다. 이 개념을 먼저 배우는 이유는 Transformer가 왜 기존의 순환 '
                            '구조를 전면 폐기하고 어텐션 메커니즘으로 대체해야만 했는지 배경을 명확히 이해할 수 있게 준비시켜 주기 때문이다.',
             'id': 'n1',
             'is_target': False,
             'layer': 0,
             'learning_points': ['순환 신경망이 시퀀스를 처리할 때 은닉 상태가 이전 단계에 종속되는 방식을 설명할 수 있다',
                                 '순차적 연산 구조로 인해 학습 시 병렬화가 불가능해지는 근본적인 원인을 파악한다'],
             'quiz': None,
             'title': 'RNN 및 시퀀스 모델링의 한계'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'Attention Mechanism', 'tier': 1, 'tier_name': '연산 / 메커니즘'},
                          {'name': 'Softmax', 'tier': 1, 'tier_name': '연산 / 메커니즘'}],
             'description': '[D2L 교재]에서는 어텐션 메커니즘이 데이터베이스의 (key, value) 쌍으로부터 데이터를 집계하는 미분 가능한 제어 '
                            '수단이라고 설명한다. 신경망이 쿼리를 사용하여 집합에서 요소를 선택하고 연관된 가중치 합을 구성하도록 돕는 핵심 원리이다. '
                            '[이 논문 PDF 원문]에서도 어텐션이 거리와 무관하게 전역 의존성을 모델링할 수 있게 해준다고 강조한다. 이 노드는 다음 '
                            '단계인 Self-Attention과 Transformer의 쿼리-키-값 내적 연산을 직관적으로 이해할 수 있는 수학적, '
                            '개념적 토대를 제공한다.',
             'id': 'n2',
             'is_target': False,
             'layer': 0,
             'learning_points': ['쿼리, 키, 값의 관계를 Nadaraya-Watson 추정기 관점에서 설명할 수 있다',
                                 '소프트맥스 정규화를 통해 가중치가 0과 1 사이의 분포로 변환되는 과정을 이해한다'],
             'quiz': None,
             'title': 'Attention Mechanism과 소프트맥스 정규화'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'Self-Attention', 'tier': 1, 'tier_name': '연산 / 메커니즘'}],
             'description': '[이 논문 PDF 원문]에 따르면 Transformer는 순환과 합성곱을 완전히 배제하고 오직 Self-Attention에만 '
                            '의존하여 입력과 출력 간의 전역 의존성을 도출한다. [D2L 교재] 및 논문의 배경에 따르면 기존 합성곱 모델은 거리 '
                            '선형/로그 비례로 연산이 증가하여 먼 위치 간의 관계 학습이 어렵지만, Self-Attention은 이를 상수 번의 연산으로 '
                            '줄인다. 이 개념은 Transformer 아키텍처의 핵심 작동 원리이며, 이후 Multi-Head Attention과 '
                            '인코더-디코더 구조를 이해하기 위한 필수 전제 조건이다.',
             'id': 'n3',
             'is_target': False,
             'layer': 1,
             'learning_points': ['Self-Attention이 입력 시퀀스 내의 토큰 간 상호작용을 상수 시간 연산으로 계산하는 방식을 설명할 수 '
                                 '있다',
                                 '거리 제한 없이 장거리 의존성을 포착할 수 있는 구조적 이점을 분석할 수 있다'],
             'quiz': None,
             'title': 'Self-Attention과 전역 의존성'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'Multi-Head Attention', 'tier': 2, 'tier_name': '레이어 / 학습 기법'}],
             'description': '[이 논문 PDF 원문]에서 Noam이 제안한 Multi-Head Attention은 단일 헤드 어텐션의 단점을 극복하고, '
                            '단일 어텐션 가중치 평균화로 인한 해상도 저하를 상쇄하기 위해 도입되었다. [발췌 4, 5]의 실험(Table 3)에 따르면 '
                            '헤드 수와 키/값 차원을 적절히 변조하여 계산량을 일정하게 유지하면서도 모델의 표현력을 극대화한다. 이 내용은 '
                            'Transformer의 핵심 구성 블록을 완성하고 후속 아키텍처 구현을 위한 중추적 역할을 한다.',
             'id': 'n4',
             'is_target': False,
             'layer': 2,
             'learning_points': ['여러 개의 어텐션 헤드가 서로 다른 부분공간에서 정보를 동시에 학습하는 구조를 이해한다',
                                 '키 차원과 헤드 수의 변화가 모델 품질과 성능에 미치는 영향을 분석할 수 있다'],
             'quiz': None,
             'title': 'Multi-Head Attention'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'Positional Encoding', 'tier': 2, 'tier_name': '레이어 / 학습 기법'},
                          {'name': 'Dropout', 'tier': 2, 'tier_name': '레이어 / 학습 기법'}],
             'description': '[이 논문 PDF 원문]과 [발췌 4, 5]에서는 순환 구조가 없는 Transformer가 시퀀스의 순서 정보를 인식하도록 '
                            '만들기 위해 사인파 기반의 Positional Encoding이나 학습된 positional embeddings를 사용함을 '
                            '보여준다. 또한 Table 3 (D) 행에서는 dropout을 통한 정규화가 과적합을 방지하는 데 매우 중요함을 증명한다. '
                            '이 노드는 모델이 순서 정보를 잃지 않고 안정적으로 학습되도록 세부 요소를 채워주며, 최종 Transformer 모델의 '
                            '완성도를 높여준다.',
             'id': 'n5',
             'is_target': False,
             'layer': 3,
             'learning_points': ['사인파 함수를 활용해 토큰의 상대적 및 절대적 위치 정보를 임베딩에 주입하는 방식을 파악한다',
                                 '드롭아웃 정규화가 대규모 트랜스포머 학습 시 과적합을 방지하는 역할을 설명할 수 있다'],
             'quiz': None,
             'title': 'Positional Encoding과 정규화 기법'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'Encoder-Decoder', 'tier': 4, 'tier_name': '구성 블록 / 소규모 아키텍처'}],
             'description': '[이 논문 PDF 원문]의 서두에 따르면 지배적인 시퀀스 변환 모델은 인코더-디코더 구성을 기반으로 하며, 최고 성능의 '
                            '모델들은 어텐션 메커니즘으로 인코더와 디코더를 연결한다. Transformer 역시 이 인코더-디코더 구성을 따르면서 내부를 '
                            '완전히 어텐션 레이어로 채웠다. 이 노드는 개별 어텐션 메커니즘과 블록들을 전체적인 시퀀스 투 시퀀스 변환 파이프라인으로 '
                            '통합하는 방법을 제공한다.',
             'id': 'n6',
             'is_target': False,
             'layer': 4,
             'learning_points': ['인코더가 입력 문장을 표현형으로 변환하고 디코더가 이를 바탕으로 출력 시퀀스를 생성하는 흐름을 추적할 수 있다',
                                 '인코더와 디코더 간의 연결 고리 역할을 하는 어텐션 블록의 배치를 이해한다'],
             'quiz': None,
             'title': 'Encoder-Decoder 구조'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'Adam', 'tier': 3, 'tier_name': '최적화 기법'}],
             'description': '[발췌 3]에 따르면 Transformer 학습에는 β1=0.9, β2=0.98, ε=10^-9 값을 가진 Adam '
                            '옵티마이저가 사용되었으며, 학습률은 웜업 스텝 구간 동안 선형 증가 후 스텝 수의 역제곱근에 비례해 감소하는 공식을 따랐다. '
                            '이 최적화 및 훈련 스케줄은 [발췌 3]의 WMT 2014 영어-독일어 및 영어-프랑스어 번역 태스크에서 안정적인 수렴과 '
                            '최고 성능을 이끌어내는 데 결정적인 역할을 했다.',
             'id': 'n7',
             'is_target': False,
             'layer': 0,
             'learning_points': ['Adam 옵티마이저의 하이퍼파라미터 설정과 웜업 스텝을 포함한 학습률 스케줄링 방식을 설명할 수 있다',
                                 '대규모 데이터셋과 하드웨어 환경에서 모델을 안정적으로 수렴시키기 위한 최적화 절차를 파악한다'],
             'quiz': None,
             'title': 'Adam 최적화와 학습 스케줄'},
            {'ai_explanation': None,
             'completed': False,
             'concepts': [{'name': 'Transformer', 'tier': 5, 'tier_name': '대규모 아키텍처'}],
             'description': '[이 논문 PDF 원문]의 최종 목표인 이 노드는 순환과 합성곱을 전혀 사용하지 않고 오직 어텐션 메커니즘만을 기반으로 하는 '
                            '새로운 네트워크 아키텍처인 Transformer를 총체적으로 이해하는 단계이다. 앞서 학습한 RNN의 한계, '
                            'Self-Attention, Multi-Head Attention, Positional Encoding, '
                            'Encoder-Decoder 구조, 그리고 Adam 학습 스케줄까지 모든 선수 지식을 결합한다. 이를 통해 WMT 2014 '
                            '영어-독일어(28.4 BLEU) 및 영어-프랑스어(41.8 BLEU) 번역 태스크에서 기존 최고 기록을 경신하고, 영어 구문 '
                            '분석 등 다른 태스크로의 일반화 가능성까지 완벽히 조망할 수 있게 된다.',
             'id': 'n8',
             'is_target': True,
             'layer': 5,
             'learning_points': ['Transformer 아키텍처가 기존 순환 신경망 대비 번역 품질과 병렬화 측면에서 우수한 이유를 종합적으로 설명할 '
                                 '수 있다',
                                 '논문에서 제시된 WMT 2014 실험 결과와 다양한 태스크로의 일반화 성능 의의를 평가할 수 있다'],
             'quiz': None,
             'title': 'Attention Is All You Need'}],
  'pdf_url': 'https://arxiv.org/pdf/1706.03762v7',
  'target_description': 'The dominant sequence transduction models are based on complex recurrent '
                        'or convolutional neural networks in an encoder-decoder configuration. The '
                        'best performing models also connect the encoder and decoder through an '
                        'attention mechanism. We propose a new simple network architecture, the '
                        'Transformer, based solely on attention mechanisms, dispensing with '
                        'recurrence and convolutions entirely. Experiments on two machine '
                        'translation tasks show these models to be superior in quality while being '
                        'more parallelizable and requiring significantly less time to train. Our '
                        'model achieves 28.4 BLEU on the WMT 2014 English-to-German translation '
                        'task, improving over the existing best results, including ensembles by '
                        'over 2 BLEU. On the WMT 2014 English-to-French translation task, our '
                        'model establishes a new single-model state-of-the-art BLEU score of 41.8 '
                        'after training for 3.5 days on eight GPUs, a small fraction of the '
                        'training costs of the best models from the literature. We show that the '
                        'Transformer generalizes well to other tasks by applying it successfully '
                        'to English constituency parsing both with large and limited training '
                        'data.',
  'target_label': 'Attention Is All You Need',
  'target_type': 'paper',
  'used_rag': True}]

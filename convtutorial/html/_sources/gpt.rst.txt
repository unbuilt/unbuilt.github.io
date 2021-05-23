

基于预训练的对话生成
======================

随着深度学习的发展，大规模预训练模型在各种任务上获得了出色的效果。在文本生成上，``GPT-2`` [GPT2]_ 和 ``GPT-3`` [GPT3]_ 展示出了很大的潜力。基于预训练的对话生成也就应运而生了。

``Dialo-GPT`` [DialoGPT]_ 扩展了 ``GPT-2`` ，用于对话生成。它把对话历史拼接后作为输入，把 ``GPT-2`` 的输出作为对话的回复。

``CDial-GPT`` 同样是基于 ``GPT-2`` ，在拼接对话历史时加入了 ``<speaker1>`` 、 ``<speaker2>`` 等标记。``CDial-GPT`` 发布了中文数据集，并提供了模型。本文在多处使用了其数据集。

``GPT-2`` 的模型并不复杂，``OpenAI`` 给出了 `代码 <https://github.com/openai/gpt-2>`_ 。
`这里 <https://github.com/karpathy/minGPT>`_ 是一个基于PyTorch的极简版本。
`Hugging Face <https://huggingface.co/>`_ 提供的 ``transformers`` 包是被广泛使用的一个 ``GPT-2`` 的实现。

``GPT-2`` 的训练需要大规模的语料和计算资源，这里就不再从头训练，直接使用 ``CDial-GPT`` 提供的对话模型来生成对话。
随后，本文基于这一基础模型，在古装影视对话数据上对模型进行微调，从而实现古装影视风格对话的生成。

基于 ``GPT`` 的对话引擎
----------------------------------

这里实现一个 ``GPTChatEngine`` 对话引擎，并利用 ``CDial-GPT`` 提供的对话模型来生成对话。

.. code-block ::

    import json
    import random
    from itertools import chain
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import OpenAIGPTLMHeadModel, GPT2LMHeadModel, BertTokenizer

    class GPTChatAgent(object):
        
        SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[speaker1]", "[speaker2]"]

        def __init__(self, model_path, min_len=1, max_len=30, temperature=0.7, top_p=0.9, top_k=0, no_sample=False, device='cpu'):
            self.min_length = min_len
            self.max_length = max_len
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k
            self.no_sample = no_sample
            self.device = device
            
            # Init models
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
            self.model = OpenAIGPTLMHeadModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
        def _encode_text(self, text):
            batch_wids = [tokenizer.encode_plus(text, max_length=100, add_special_tokens=True)["input_ids"]]
            with torch.no_grad():
                bert_result = self.bert_model(torch.LongTensor(batch_wids))
                return bert_result[1][0].numpy()

        def _search(self, message):
            encoded_message = self._encode_text(message)
            results = self.index.search(np.array([encoded_message]), self.limit)
            
            return results
        
        def _top_filtering(self, logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
            """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
                Args:
                    logits: logits distribution shape (vocabulary size)
                    top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
                    top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                        whose total probability mass is greater than or equal to the threshold top_p.
                        In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                        the threshold top_p.
                    threshold: a minimal threshold to keep logits
            """
            assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
            top_k = min(top_k, logits.size(-1))
            if top_k > 0:
                # Remove all tokens with a probability less than the last token in the top-k tokens
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = filter_value

            if top_p > 0.0:
                # Compute cumulative probabilities of sorted tokens
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probabilities > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Back to unsorted indices and set them to -infinity
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = filter_value

            indices_to_remove = logits < threshold
            logits[indices_to_remove] = filter_value

            return logits
        
        def _build_input_from_segments(self, history, reply, with_eos=True):
            """ Build a sequence of input from 3 segments: persona, history and last reply """
            bos, eos, pad, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(CDialGPTChatAgent.SPECIAL_TOKENS)
            sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
            sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
            instance = {}
            instance["input_ids"] = list(chain(*sequence))
            instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:])
                                                for _ in s]
            return instance, sequence
        
        def _sample_sequence(self, history, current_output=None):
            special_tokens_ids = self.tokenizer.convert_tokens_to_ids(CDialGPTChatAgent.SPECIAL_TOKENS)
            if current_output is None:
                current_output = []

            for i in range(self.max_length):
                instance, sequence = self._build_input_from_segments(history, current_output, with_eos=False)
                input_ids = torch.tensor(instance["input_ids"], dtype=torch.long, device=self.device).unsqueeze(0)
                token_type_ids = torch.tensor(instance["token_type_ids"], dtype=torch.long, device=self.device).unsqueeze(0)

                logits, *_ = self.model(input_ids, token_type_ids=token_type_ids)
                logits = logits[0, -1, :] / self.temperature
                logits = self._top_filtering(logits, top_k=self.top_k, top_p=self.top_p)
                probs = F.softmax(logits, dim=-1)

                prev = torch.topk(probs, 1)[1] if self.no_sample else torch.multinomial(probs, 1)
                if i < self.min_length and prev.item() in special_tokens_ids:
                    while prev.item() in special_tokens_ids:
                        prev = torch.multinomial(probs, num_samples=1)

                if prev.item() in special_tokens_ids:
                    break
                current_output.append(prev.item())

            return current_output
            
        def reply(self, message):
            with torch.no_grad():
                history = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(message))]
                out_ids = self._sample_sequence(history)
                out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
                return out_text

我们用 ``CDial-GPT`` 的模型初始化这个对话引擎。

.. code-block ::

    class ConversationSystem(object):
                
        # ...

        def __init__(self):
            self.chat_engines = [
                TemplateChatAgent(ConversationSystem.patterns),
                CDialGPTChatAgent(r'\CDial-GPT\models'),
                TemplateChatAgent(ConversationSystem.fallback_patterns),
            ]

        # ...


    conv_system = ConversationSystem()
    conv_system.interact_cli()

在命令行与对话系统交互，可能得到类似下面的对话。

::

    User:你好
    AI: 你好
    User:你过来
    AI: 我 在 家
    User:你要到哪里去
    AI: 还 在 考 虑
    User:快回去
    AI: 回 哪 里
    User:回你该去的地方
    AI: 没 去 过


古装影视风格对话生成模型
---------------------------

在小数据集上微调是预训练模型的主要使用方法。这里在 ``CDial-GPT`` 模型的基础上，用古装影视对话数据进行微调，从而可以生成风格明显的对话。

获取数据
^^^^^^^^^^^^

通过收集古装影视台词，我们得到如下的对话数据，并保存在 ``tv.json`` 文件中。

::

    [
        "华妃娘娘请皇上移步翊坤宫用膳",
        "朕不过去了"
    ],
    [
        "朕不过去了",
        "那皇上的意思是"
    ],
    [
        "那皇上的意思是",
        "朕去瞧瞧皇后"
    ]



微调训练
^^^^^^^^^^^^

微调训练需要首先获取 ``CDial-GPT`` 的源代码，并安装相关环境。
然后运行下面的命令训练模型。

::

    python train.py --pretrained --model_checkpoint ./models/ --data_path ./data/tv.json --scheduler linear

模型训练完成后，我们在对话系统中使用这个模型。

.. code-block ::

    class ConversationSystem(object):
                
        # ...

        def __init__(self):
            self.chat_engines = [
                TemplateChatAgent(ConversationSystem.patterns),
                CDialGPTChatAgent(r'\CDial-GPT\model_tv'),
                TemplateChatAgent(ConversationSystem.fallback_patterns),
            ]

        # ...


    conv_system = ConversationSystem()
    conv_system.interact_cli()

在命令行与对话系统交互，可能得到类似下面的对话。

::

    User:你在干什么
    AI: 我 在 陪 小 主 说 话
    User:快去干活
    AI: 等 一 下 再 来
    User:少要多嘴
    AI: 王 爷 有 什 么 难 言 之 隐
    User:休得无理
    AI: 奴 婢 知 道 错 了

从上面的例子可以看出，微调后的模型对话风格还是非常明显的。
``Controllable Neural Text Generation`` [Controllable]_ 这篇文章也介绍了控制文本生成的方法。

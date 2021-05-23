基于检索的方法
===============

基于检索的对话生成方法借鉴了信息检索技术，把对话的生成任务改成了从已有的对话库中找出相近的对话作为答案。

基本的基于检索的对话引擎
--------------------------

构建一个基于检索的对话引擎，需要以下工作：

1. 收集并整理对话数据
2. 对对话数据进行索引
3. 根据用户输入检索出多个候选问答对
4. 对候选回复进行排序，输出回答

准备对话数据
^^^^^^^^^^^^^

目前常用的对话数据的收集方式主要是从通过从论坛、社交媒体等网站爬取用户的发言或者从影视剧中提取台词。
收集到对话数据，需要进行一系列的清洗以提高数据的质量。

数据收集用到的技术如网络爬虫。

这里我们直接使用已经公开的数据集。`CDial-GPT` 开放了较高质量的对话数据，每一个对话片段包含多轮。目前我们只考虑单轮的对话，所以可以把多轮对话处理成一组单轮对话，同时我们过滤掉太长的句子，以及回答对中问答太短的句子。

.. code-block :: 

    # 处理对话数据
    def process_corpus(datapath, output_path):
        f = open(datapath, 'r', encoding='utf-8')
        convs = json.load(f)
        f.close()

        qas = []
        for conv in convs:
            qas.extend(zip(conv[:-1], conv[1:]))
        
        qas = [(q, a) for q, a in qas if len(q) < 30 and len(a) < 30 and len(q) > 4]

        f = open(output_path, 'w', encoding='utf-8')
        f.write(json.dumps(qas, ensure_ascii=False))
        f.close()


经过上述的处理，我们就得到很多问答对。下一步就是对这些回答对进行索引。

对对话数据建立索引
^^^^^^^^^^^^^^^^^^^^^^

我们可以自己建立索引并编写检索算法，也可以利用现有的搜索引擎系统，比如 ``ElasticSearch`` 等。这里我们使用基于 ``Python`` 的 ``Ｗhoosh`` 搜索引擎模块。
``Whoosh`` 建立索引的时候需要一个分词器，我们使用基于 ``Python`` 的 ``jieba`` 分词模块。

.. code-block ::

    from whoosh.index import create_in
    from whoosh.fields import Schema, TEXT, ID
    from jieba.analyse import ChineseAnalyzer

    def build_index(datapath, index_dir):
        f = open(datapath, 'r', encoding='utf-8')
        qas = json.load(f)
        f.close()

        analyzer = ChineseAnalyzer()
        
        schema = Schema(
            context=TEXT(stored=True, analyzer=analyzer),
            response=TEXT(stored=True))

        chat_idx = create_in(index_dir, schema)
        writer = chat_idx.writer()
        for q, a in qas:
            writer.add_document(
                context=q,
                response=a)
        writer.commit()


根据用户输入检索出多个相关问答对
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

创建好索引之后，我们就可以加载索引并进行检索，这就是 ``RetrievalChatAgent`` 。

.. code-block ::

    import random
    from whoosh.query import Term
    from whoosh.qparser import QueryParser, MultifieldParser
    from whoosh.index import open_dir
    from jieba.analyse import ChineseAnalyzer

    class RetrievalChatAgent(object):

        def __init__(self, index_dir, limit=5):
            self.idx = open_dir(index_dir)
            self.searcher = self.idx.searcher()
            self.parser = QueryParser("context", schema=self.idx.schema)
            self.limit = limit

        def reply(self, message):
            query = self.parser.parse(message)
            results = self.searcher.search(query, limit=self.limit)
            cands = [res['response'] for res in results]
            if cands:
                return random.choice(cands)
            
            return ''


检索的过程是用输入与索引中的问题进行匹配，返回匹配度最高的问答对。``Ｗhoosh`` 检索时使用的匹配方法是 ``BM25`` 。

.. code-block ::

    class ConversationSystem(object):
        
        patterns = [
            (r'你好[吗啊呀]?', {'*': ['你好', '嗨']}),
            (r'你是男是女', {'*': ['你觉得呢', '你呢']}),
            (r'你是谁', {'*': ['我是一个AI', '我是个机器人']}),
            (r'你(多大|几岁)[吗啊呀]?', {'*': ['我还年轻', '这是个秘密']}),
            (r'你在(哪里|哪儿|什么地方)', {'*': ['我在云端，你在哪儿', '我在你的身边，你在哪里呢']}),
            (r'我是(?P<Gender>[男女])(的|生|人)', {'男': ['帅哥好'], '女': ['美女好']}),
            (r'我也?在(?P<Place>.+)', {'北京': ['北京现在很冷吧'], '*': ['你在{Place}？']})
        ]

        fallback_patterns = [
            (r'.*', {'*': ['你说什么', '不好意思，没明白你的话']})
        ]

        def __init__(self):
            self.chat_engines = [
                TemplateChatAgent(ConversationSystem.patterns),
                RetrievalChatAgent(''),
                TemplateChatAgent(ConversationSystem.fallback_patterns),
            ]

        def reply(self, uid, message):
            result = ''
            for chat_engine self.chat_engines:
                result = chat_engine.reply(message)
                if result:
                    break
            
            return result
        
        def interact_cli(self):
            while True:
                query = input('User:')
                if query == 'Q':
                    break
                print('AI:', self.reply('UserA', query))

    conv_system = ConversationSystem()
    conv_system.interact_cli()



运行后可以得到下面的对话。可以看出，对话系统可以开始天南海北的瞎聊了。

::

    User: 你好
    AI: 嗨
    User: 你是谁
    AI: 我是个机器人
    User: 你喜欢什么运动
    AI: 麻烦 出个 选择题 ~ 好哇 ? > ~
    User: 你喜欢运动吗
    AI: 逛街 ， 吃饭 ， 简称 ： 逛 吃 逛 吃 逛 吃
    User: 除了吃就知道吃
    AI: 我 就是 这么 没出息 啊
    User: 给AI丢脸
    AI: 你说什么
    User: 说你笨
    AI: 真的 吗
    User: 今天天气不错
    AI: 是 的 有点 热
    User: 我想去吃火锅
    AI: 我 想 在家 吃火锅


这里取到检索结果后，直接随机选取了一个回复作为输出。一种更好的方法是对初步检索出的结果作进一步排序后再选择一个作为回复。

候选回复对进行排序
^^^^^^^^^^^^^^^^^^^^^^^^

排序可以使用各种文本匹配的方法，除 ``BM25`` 外，还有编辑距离、最长公共字串、词向量相似度等很多方法。文本匹配是自然语言处理中的基本问题。

这里既可以基于用户输入对索引问答对中的问题进行排序，也可以基于用户的输入对问答对中的答案进行排序并选择。
`Deep Chit-Chat: Deep Learning for ChatBots` [Tutorial]_ 介绍了深度学习在聊天机器人中的应用，特别是各种答案匹配的方法。


基于语义的索引及检索
---------------------

上面的检索系统在索引问答对时使用的是词，所以在检索时也只能找出包含输入中的词的问答对。对于语义相同但词语不同的句子并不能检索出来。
比如输入 ``我很伤心``，是不能检索出 ``我很难过`` 。

基于语义的检索可以解决这个问题。语义检索系统索引和检索的对象是向量，需要在索引前对问答对进行编码转换成一个向量，在检索前把输入编码为一个向量。由于语义相似的词的向量相近，所以可以检索出字面不同但意思相近的问答对。

除了一开始的收集问答对数据以及最后的排序外，基于语义的检索需要做以下工作：

1. 对问答对进行编码，把问答转换成一个向量
2. 对向量进行索引
3. 在检索时，先对输入编码为向量
4. 对向量进行检索，得到候选问答对
5. 对候选回复进行排序，输出回答

下面实现一个 ``SemanticChatAgent``，使用 ``Transformer`` 包里的 ``BertModel`` 对句子进行编码，得到表示句子语义的向量。
可以对向量进行索引及检索的工具有 ``KD-Tree``, `annoy`_ 等，这里使用 `faiss`_ 进行语义索引及查询。
问答数据使用前面已经处理好的问答对。

.. code-block ::

    import json
    import random
    import numpy as np
    import torch 
    from transformers import BertTokenizer, BertModel
    from tqdm import tqdm

    import faiss

    class SemanticChatAgent(object):

        def __init__(self, filename, limit=10):
            self.limit = limit
            
            # Init models
            model_name = 'bert-base-chinese'
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.bert_model = BertModel.from_pretrained(model_name)
            self.bert_model.eval()
            
            # Load data
            f = open(filename, 'r', encoding='utf-8')
            self.qas = json.load(f)
            f.close()
            
            # Encode data
            encoded_qs = []
            for q, a in tqdm(qas):
                encoded_qs.append(self._encode_text(q))

            # Build index
            self.dimension = 768
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array(encoded_qs))
            
        def _encode_text(self, text):
            batch_wids = [tokenizer.encode_plus(text, max_length=100, add_special_tokens=True)["input_ids"]]
            with torch.no_grad():
                bert_result = self.bert_model(torch.LongTensor(batch_wids))
                return bert_result[1][0].numpy()

        def _search(self, message):
            encoded_message = self._encode_text(message)
            results = self.index.search(np.array([encoded_message]), self.limit)
            
            return results
            
        def reply(self, message):
            results = self._search(message)
            candidates = [self.qas[idx][1] for idx in results[1][0]]
            
            return random.choice(candidates)


实例化一个 ``SemanticChatAgent``，输入 ``我好难过``，可能会得到 ``快 来 抱抱 我`` 、``不要 难过 ， 不要 烦`` 等回复。

.. code-block ::

    agent = SemanticChatAgent('C:\Works\data\seq2seq\LCCC-base_train.seq2seq.json')
    agent.reply('我好难过')

如果查看搜索结果的话，可以发现不仅检索出了 ``难过``，还找到了包含 ``不开心`` 、 ``真伤感`` 等词的句子。这就是基于语义检索的优势。
同时也可以看到，语义检索在提高召回率的同时，也降低了精确率，所以基于用户输入对回答句子进行排序的功能就很重要了。

::

    9.487286 ['只是 好 难过', '心塞 的 时间 已经 过 了']
    10.490826 ['我 不 开心', '快 来 抱抱 我']
    10.844768 ['心疼 脸上 的 伤', '小 团子 那个 好像 是 痘痘 哈哈哈哈']
    11.026273 ['我 想 多 了', '加班加点 冲锋陷阵 年终 决算 都 半夜三更 走']
    11.116728 ['其实 很 难过', '不要 难过 ， 不要 烦']
    11.802321 ['妹妹 真 伤感', '对不起']
    12.01674 ['今天 放 的 我 好 难过', '我 已经 没看 了 ， 知道 后面 会虐 ， 不想 看 了']
    12.237868 ['嗯 ， 我 生气 我 难过', '摸摸 ， 我们 要 让 自己 变得 强大']
    13.057215 ['我 快乐 好久 了', '那 是因为 你 没 回去 吧 ？']
    13.074891 ['想太多 了', '那 就是 伤不起']

    
.. _`annoy`: https://github.com/spotify/annoy
.. _`faiss`: https://github.com/facebookresearch/faiss


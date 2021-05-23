
序列到序列生成
=================

`Sequence to Sequence Learning with Neural Networks` [Seq2Seq]_ 这篇文章描述了序列到序列模型，它读入一个输入句子，先通过循环神经网络编码成一个隐式的向量，再根据这个隐式的向量使用循环神经网络语言模型输出一个句子。这个模型在机器翻译问题上取得了很好的效果，也被用在于诸如图片描述生成、问答、对话生成等任务上。

循环神经网络
------------

循环神经网络是神经元可以接受自身信息从而形成环路的网络结构，能够处理像自然语言中的句子等任意长度的序列数据。

``神经网络与深度学习`` [NNDL]_ 这本书讲解了神经网络与深度学习技术的基本原理，其中包括了循环神经网络等基础的神经网络模型。

`The Unreasonable Effectiveness of Recurrent Neural Networks <http://karpathy.github.io/2015/05/21/rnn-effectiveness/>`_ 
和 `Understanding LSTM Networks <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_
这两篇文章是学习循环神经网络的很好的参考。


基于LSTM的序列到序列对话生成
------------------------------

根据序列到序列模型，实现对话生成需要如下步骤：

1. 数据预处理：把对话数据处理成QA对。
2. 建立词典：后续模型学习需要使用数字代替词，这一步根据QA对建立词典，词典为每个词生成一个唯一的数字，并提供词到数字的相互转换功能。
3. 实现模型：使用Pytorch实现编码器模型及解码器模型。
4. 训练模型：使用QA训练模型。
5. 生成对话：使用训练好的模型生成对话回复。

这里来实现一个基于 ``LSTM`` 的序列到序列对话生成模型。本节大量参考了 ``PyTorch`` 的 ``Chatbot Tutorial`` [Chatbot]_ 。

数据预处理
^^^^^^^^^^^^^^

这里依旧使用 ``CDial-GPT`` 开放的数据集作为生成模型的训练数据。首先处理原始数据为问答对。

.. code-block :: 

    def process_corpus(datapath, output_path):
        f = open(datapath, 'r', encoding='utf-8')
        convs = json.load(f)
        f.close()

        qas = []
        for conv in tqdm(convs):
            qas.extend(zip(conv[:-1], conv[1:]))
        
        qas = [(q, a) for q, a in qas if len(q) < 30 and len(a) < 30 and len(q) > 4]

        f = open(output_path, 'w', encoding='utf-8')
        f.write(json.dumps(qas, ensure_ascii=False))
        f.close()


处理数据后，保存到文件中备用，并重新读入。

.. code-block :: 

    process_corpus('\data\LCCC\LCCC-base-split\LCCC-base_train.json', '\data\seq2seq\LCCC-base_train.seq2seq.json')

    f = open('\data\seq2seq\LCCC-base_train.seq2seq.json', 'r', encoding='utf-8')
    qas = json.load(f)
    f.close()


问答对内容是这样的：

.. code-block :: 

    qas[:10]

    [['道歉 ！ ！ 再有 时间 找 你 去', '领个 搓衣板 去 吧'],
    ['咬咬牙 这回 要 全入 了 ！', '干完 这 一票 我 的 会员 等级 就要 升 了 ！'],
    ['干完 这 一票 我 的 会员 等级 就要 升 了 ！', '升 了 继续'],
    ['代表 了 哪里 的 普通人 ？', '我 瞎说 的'],
    ['早点 好 起来 啊 。 生日快乐', '好 得 差不多 啦'],
    ['好 得 差不多 啦', '那 很 好 啊'],
    ['那么 早 ！ ！ 我 的 考试 周 还 没有 开始 呢 ！', '是 啊 ， 今年 好 快 啊'],
    ['是 啊 ， 今年 好 快 啊', '不止 今年 ， 我 发现 你 每次 都 好 早'],
    ['不止 今年 ， 我 发现 你 每次 都 好 早', '小心 乌鸦嘴 ， 下 一次 就 最晚 了'],
    ['今天 不 知道 能下 么 ， 不过 天 又 冷 了', '希望 能 …']]


词典及输入输出
^^^^^^^^^^^^^^^^^

首先编写一个 ``Voc`` 来生成词典。

.. code-block ::

    class Voc:

        PAD_TOKEN = "PAD"
        SOS_TOKEN = "SOS"
        EOS_TOKEN = "EOS"

        PAD_TOKEN_IDNEX = 0
        SOS_TOKEN_INDEX = 1
        EOS_TOKEN_INDEX = 2

        def __init__(self, name):
            self.name = name
            self.trimmed = False
            self.word2index = {}
            self.word2count = {}
            self.index2word = {
                Voc.PAD_TOKEN_IDNEX: Voc.PAD_TOKEN,
                Voc.SOS_TOKEN_INDEX: Voc.SOS_TOKEN,
                Voc.EOS_TOKEN_INDEX: Voc.EOS_TOKEN
            }
            self.num_words = len(self.index2word)

        def add_sentence(self, sentence):
            for word in sentence.split(' '):
                self.add_word(word)
        
        def add_word(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.num_words
                self.word2count[word] = 1
                self.index2word[self.num_words] = word
                self.num_words += 1
            else:
                self.word2count[word] += 1
        
        def trim(self, min_count):
            if self.trimmed:
                return
            self.trimmed = True

            keep_words = []
            for k, v in self.word2count.items():
                if v >= min_count:
                    keep_words.append(k)
            
            print('keep_words {} / {} = {:.4f}'.format(
                len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
            ))

            # Reinitialize dictionaries
            self.word2index = {}
            self.word2count = {}
            self.index2word = {
                Voc.PAD_TOKEN_IDNEX: Voc.PAD_TOKEN,
                Voc.SOS_TOKEN_INDEX: Voc.SOS_TOKEN,
                Voc.EOS_TOKEN_INDEX: Voc.EOS_TOKEN
            }
            self.num_words = len(self.index2word)

            for word in keep_words:
                self.add_word(word)


使用问答对数据，生成词典。这里只保留在问答对中出现过2次以上的词。

.. code-block ::

    voc = Voc('LCCC-base')
    for q, a in tqdm(qas):
        voc.add_sentence(q)
        voc.add_sentence(a)

    MIN_COUNT = 2
    voc.trim(MIN_COUNT)


根据过滤后的词典，再次处理问答对，去掉那些含有不在词典中的词的句子。

.. code-block ::

    keep_qas = []
    for q, a in tqdm(qas):
        keep_q = True
        keep_a = True
        for word in q.split():
            if word not in voc.word2index:
                keep_q = False
                break
        for word in a.split():
            if word not in voc.word2index:
                keep_a = False
                break
        if keep_q and keep_a:
            keep_qas.append((q, a))
        
    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(qas), len(keep_qas), len(keep_qas) / len(qas)))


实现几个辅助函数，用于转换数据。

.. code-block ::

    def words_to_indexes(voc, words):
        return [voc.word2index[word] for word in words] + [Voc.EOS_TOKEN_INDEX]

    def zero_padding(l, fillvalue=Voc.PAD_TOKEN_IDNEX):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def binary_matrix(l, value=Voc.PAD_TOKEN_IDNEX):
        m = []
        for seq in l:
            mask = [0 if token == Voc.PAD_TOKEN_IDNEX else 1 for token in seq]
            m.append(mask)
        return m

    def input_var(l, voc):
        indexes_batch = [words_to_indexes(voc, sentence.split()) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        pad_list = zero_padding(indexes_batch)
        pad_var = torch.LongTensor(pad_list)

        return pad_var, lengths

    def output_var(l, voc):
        indexes_batch = [words_to_indexes(voc, sentence.split()) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        pad_list = zero_padding(indexes_batch)

        mask = binary_matrix(pad_list)
        mask = torch.BoolTensor(mask)
        pad_var = torch.LongTensor(pad_list)

        return pad_var, mask, max_target_len

    def batch_train_data(voc, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0].split()), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        
        inp, lengths = input_var(input_batch, voc)
        outp, mask, max_target_len = output_var(output_batch, voc)

        return inp, lengths, outp, mask, max_target_len


模型
^^^^

模型包括编码器，注意力机制，以及解码器。

编码器
""""""""

.. code-block ::

    class EncoderRNN(torch.nn.Module):

        def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
            super(EncoderRNN, self).__init__()
            self.n_layers = n_layers
            self.hidden_size = hidden_size
            self.embedding = embedding

            self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

        def forward(self, input_seq, input_lengths, hidden=None):
            embeded = self.embedding(input_seq)
            packed = torch.nn.utils.rnn.pack_padded_sequence(embeded, input_lengths)
            outputs, hidden = self.gru(packed, hidden)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

            return outputs, hidden


注意力
""""""

.. code-block ::

    class Attn(torch.nn.Module):

        def __init__(self, method, hidden_size):
            super(Attn, self).__init__()
            self.method = method
            if self.method not in ['dot', 'general', 'concat']:
                raise ValueError(self.method, "is not an appropriate attention method.")
            self.hidden_size = hidden_size
            if self.method == 'general':
                self.attn = torch.nn.Linear(self.hidden_size, self.hidden_size)
            elif self.method == 'concat':
                self.attn = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
                self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))
            
        def _dot_score(self, hidden, encoder_output):
            return torch.sum(hidden * encoder_output, dim=2)
        
        def _general_score(self, hidden, encoder_output):
            energy = self.attn(encoder_output)
            return torch.sum(hidden * energy, dim=2)
        
        def _concat_score(self, hidden, encoder_output):
            energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
            return torch.sum(self.v * energy, dim=2)

        def forward(self, hidden, encoder_outputs):
            # Calculate the attention weights (energies) based on the given method
            if self.method == 'general':
                attn_energies = self._general_score(hidden, encoder_outputs)
            elif self.method == 'concat':
                attn_energies = self._concat_score(hidden, encoder_outputs)
            elif self.method == 'dot':
                attn_energies = self._dot_score(hidden, encoder_outputs)

            # Transpose max_length and batch_size dimensions
            attn_energies = attn_energies.t()

            # Return the softmax normalized probability scores (with added dimension)
            return F.softmax(attn_energies, dim=1).unsqueeze(1)

解码器
""""""

.. code-block ::

    class LuongAttnDecoderRNN(torch.nn.Module):

        def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
            super(LuongAttnDecoderRNN, self).__init__()

            self.attn_model = attn_model
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.n_layers = n_layers
            self.dropout = dropout

            self.embedding = embedding
            self.embedding_dropout = torch.nn.Dropout(dropout)
            self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=False)
            self.concat = torch.nn.Linear(hidden_size * 2, hidden_size)
            self.out = torch.nn.Linear(hidden_size, output_size)

            self.attn = Attn(attn_model, hidden_size)

        
        def forward(self, input_step, last_hidden, encoder_outputs):
            embedded = self.embedding(input_step)
            embedded = self.embedding_dropout(embedded)

            rnn_output, hidden = self.gru(embedded, last_hidden)
            attn_weights = self.attn(rnn_output, encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

            rnn_output = rnn_output.squeeze(0)
            context = context.squeeze(1)

            concat_input = torch.cat((rnn_output, context), 1)
            concat_output = torch.tanh(self.concat(concat_input))

            output = self.out(concat_output)
            output = F.softmax(output, dim=1)

            return output, hidden


训练模型
^^^^^^^^

计算损失

.. code-block ::

    def mask_nll_loss(inp, target, mask):
        total = mask.sum()
        cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = cross_entropy.masked_select(mask).mean()
        loss = loss.to(device)

        return loss, total.item()

训练函数

.. code-block ::

    def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=30):
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_variable = input_variable.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)

        lengths = lengths.to('cpu')

        loss = 0
        print_losses = []
        totals = 0

        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        decoder_input = torch.LongTensor([[Voc.SOS_TOKEN_INDEX for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        decoder_hidden = encoder_hidden[:decoder.n_layers]

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = target_variable[t].view(1, -1)
                mask_loss, total = mask_nll_loss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                totals += total

                print_losses.append(mask_loss.item() * total)
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)

                mask_loss, total = mask_nll_loss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                totals += total

                print_losses.append(mask_loss.item() * total)
        
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

        # Adjust model weights
        encoder_optimizer.step()
        decoder_optimizer.step()

        return sum(print_losses) / totals

训练

.. code-block ::

    model_name = 's2s_model'
    attn_model = 'dot'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 128

    embedding = torch.nn.Embedding(voc.num_words, hidden_size)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 60000
    print_every = 1000
    save_every = 10000
    
    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # If you have cuda, configure cuda to call
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    
    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()   

    training_batches = [batch_train_data(voc, [random.choice(keep_qas) for _ in range(batch_size)]) for _ in range(n_iteration)]

    save_dir = os.path.join("data", "save")
    corpus_name = 'LCCC-base'

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    
    # Training loop
    print("Training...")
    for iteration in tqdm(range(start_iteration, n_iteration + 1)):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
    
        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss
    
        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0
    
        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

训练好后，模型就保存在了文件里。


对话生成
^^^^^^^^^^

我们使用前面训练好的模型，实现一个 ``Seq2SeqChatAgent`` 对话引擎。

.. code-block ::

    import random
    import os
    import torch

    class Seq2SeqChatAgent(object):

        def __init__(self, model_filename, device):
            self.hidden_size = 500
            self.dropout = 0.1
            self.encoder_n_layers = 2
            self.decoder_n_layers = 2
            self.attn_model = 'dot'
            self.max_length = 30
            self.device = device
            checkpoint = torch.load(model_filename)
            self.encoder_sd = checkpoint['en']
            self.decoder_sd = checkpoint['de']
            self.encoder_optimizer_sd = checkpoint['en_opt']
            self.decoder_optimizer_sd = checkpoint['de_opt']
            embedding_sd = checkpoint['embedding']
            self.voc = Voc('LCCC-base')
            self.voc.__dict__ = checkpoint['voc_dict']

            self.embedding = torch.nn.Embedding(self.voc.num_words, self.hidden_size)
            self.embedding.load_state_dict(embedding_sd)

            self.encoder = EncoderRNN(self.hidden_size, self.embedding, self.encoder_n_layers, self.dropout)
            self.decoder = LuongAttnDecoderRNN(self.attn_model, self.embedding, self.hidden_size, self.voc.num_words, self.decoder_n_layers, self.dropout)

            self.encoder.load_state_dict(self.encoder_sd)
            self.decoder.load_state_dict(self.decoder_sd)

            self.encoder = self.encoder.to(device)
            self.decoder = self.decoder.to(device)

            # Set dropout layers to eval mode
            self.encoder.eval()
            self.decoder.eval()

        def reply(self, message):
            input_sentence = [x for x in jieba.cut(message)]
            # words -> indexes
            indexes_batch = [words_to_indexes(self.voc, input_sentence)]
            # Create lengths tensor
            lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
            # Transpose dimensions of batch to match models' expectations
            input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
            # Use appropriate device
            input_batch = input_batch.to(self.device)
            #lengths = lengths.to(device)
            
            # Decode sentence with searcher
            with torch.no_grad():
                tokens, scores = self._greedy_search(input_batch, lengths, self.max_length)
                # indexes -> words
                decoded_words = [self.voc.index2word[token.item()] for token in tokens]
                # Format response sentence
                decoded_words[:] = [x for x in decoded_words if not (x == 'EOS' or x == 'PAD')]
                return ''.join(decoded_words)

        def _greedy_search(self, input_seq, input_length, max_length):
            # Forward input through encoder model
            encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
            # Prepare encoder's final hidden layer to be first hidden input to the decoder
            decoder_hidden = encoder_hidden[:self.decoder.n_layers]
            # Initialize decoder input with SOS_token
            decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * Voc.SOS_TOKEN_INDEX

            # Initialize tensors to append decoded words to
            all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
            all_scores = torch.zeros([0], device=self.device)
            # Iteratively decode one word token at a time
            for _ in range(max_length):
                # Forward pass through decoder
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Obtain most likely word token and its softmax score
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

                # Record token and score
                all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
                all_scores = torch.cat((all_scores, decoder_scores), dim=0)
                # Prepare current token to be next decoder input (add a dimension)
                decoder_input = torch.unsqueeze(decoder_input, 0)

            # Return collections of word tokens and scores
            return all_tokens, all_scores

我们使用基于生成模型的对话引擎进行交互。

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
        checkpoint_iter = 60000
        checkpoint_filename = os.path.join(save_dir, model_name, corpus_name,
            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
            '{}_checkpoint.tar'.format(checkpoint_iter))

        self.chat_engines = [
            TemplateChatAgent(ConversationSystem.patterns),
            Seq2SeqChatAgent(checkpoint_filename),
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



运行后可以得到下面的对话。

::

    User:你好
    AI: 你好
    User:昨天下了一场大雨
    AI: 下大雨了
    User:英超结束了
    AI: 结束了
    User:明天太阳会不会出来
    AI: 不会
    User:夏天来了
    AI: 夏天夏天夏天夏天夏天夏天夏天就来了


解码采样策略
^^^^^^^^^^^^^^^^^

``Seq2SeqChatAgent`` 在生成句子时每一步都取概率最大的词，这种选择目标词的方式称为 ``贪心搜索`` 。这种选择方式只会产生一个生成结果。
如果在每一步多保留几个候选词，有可能会生成出更好的句子，或者使生成的结果具有多样性。这种选择目标词的方式称为 ``Beam Search`` 。

.. code-block ::

    class BeamSearchDecoder(nn.Module):

        def __init__(self, encoder, decoder, k):
            super(BeamSearchDecoder, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.k = k
        
        def forward(self, input_seq, input_length, max_length):
            encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
            decoder_hidden = encoder_hidden[:decoder.n_layers]
            decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * Voc.SOS_TOKEN_INDEX
            candidates = [{
                'Hidden': decoder_hidden,
                'Inputs': decoder_input,
                'Tokens': [], 
                'Scores': []
            }]


            dones = []
            for _ in range(max_length):
                next_candidates = []
                for candidate in candidates:
                    current_input = candidate['Inputs']
                    if current_input == Voc.EOS_TOKEN_INDEX:
                        dones.append(candidate)
                        continue
                    current_hidden = candidate['Hidden']
                    decoder_output, decoder_hidden = self.decoder(current_input, current_hidden, encoder_outputs)
                    c_scores, c_input = torch.topk(decoder_output, self.k, dim=1)
                    for s, i in zip(c_scores[0], c_input[0]):
                        new_score = candidate['Scores'] + [s]
                        new_tokens = candidate['Tokens'] + [i]
                        
                        next_candidates.append({
                            'Hidden': decoder_hidden,
                            'Inputs': i.unsqueeze(0).unsqueeze(0),
                            'Tokens': new_tokens, 
                            'Scores': new_score
                        })

                next_candidates.sort(key=lambda e: sum(e['Scores'])/(len(e['Scores']))*len(set(e['Tokens']))*len(set(e['Tokens']))/len(e['Tokens'])/len(e['Tokens']), reverse=True)
                if self.k <= len(dones):
                    break
                candidates = next_candidates[:self.k-len(dones)]
            
            for candidate in candidates:
                current_input = candidate['Inputs']
                if current_input == Voc.EOS_TOKEN_INDEX:
                    dones.append(candidate)

            candidates = dones
            candidates.sort(key=lambda e: sum(e['Scores'])/(len(e['Scores']))*len(set(e['Tokens']))*len(set(e['Tokens']))/len(e['Tokens'])/len(e['Tokens']), reverse=True)  

            if candidates:
                candidate = random.choice(candidates)
                r = torch.stack(candidate['Tokens'])
                s = torch.stack(candidate['Scores'])

                return r, s
        
            return None, None


``Controllable Neural Text Generation`` [Controllable]_ 这篇文章介绍了各种解码策略。

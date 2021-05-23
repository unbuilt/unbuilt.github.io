基于脚本的方法
===============

基于脚本的方法也可以叫作基于模板或者规则的方法。脚本系统包括一个脚本引擎及其支持的脚本语言，以及由人工编写一系列模板（或者规则）。接受用户输入后，脚本引擎拿用户输入与模板相匹配，匹配到有效的模板后，根据模板指定的方式产生回答。

通常脚本引擎支持在模板中对输入输出的处理：

* 根据输入匹配到一个模板
* 根据模板的规则从输入中提取内容
* 对输入及提取的内容作替换、添加等操作
* 映射到一个回答
* 替换回答中的占位符
* 返回处理好的回答

对于脚本引擎，从简单的Eliza到AIML等很多功能强大的开源项目，我们有众多选择。

在了解这些功能强大的脚本引擎之前，让我们尝试实现一些最简单的基于脚本的对话引擎。

实现简单的脚本系统
--------------------

* 价值一亿的人工智能程序
* 基于正则匹配的脚本引擎
* 带模板的脚本引擎

价值几亿的人工智能核心代码
""""""""""""""""""""""""""""

有一段号称价值几亿的人工智能核心代码曾流传甚广，它实现了一个简单的脚本引擎，不需要编写人工规则，当然也没定义编写规则需要的脚本语言。

.. code::

   class JokeChatAgent(object):

       def reply(self, message):
           return message.replace('吗', '').replace('？', '！')


如果我们在前面的对话系统中使用这个脚本引擎并运行，会得到这样的结果：

::

    User: 你好
    AI: 你好
    User: 吃饭了吗？
    AI: 吃饭了！
    User: 开心吗
    AI: 开心
    User: Q


基于正则匹配的脚本引擎
""""""""""""""""""""""

基于正则表达式的脚本引擎是最常见的形式，在很多对话产品中都会使用正则表达式匹配用户输入，如果匹配成功，则输出对应回答。

下面是一个简单的正则表达式脚本引擎的实现。

.. code::

    import re
    import random

    class RegexChatAgent(object):

        def __init__(self, patterns):
            self.patterns = [(re.compile(regex), answers) for regex, answers in patterns]
        
        def reply(self, message):
            for regex, answers in self.patterns:
                if regex.match(message):
                    return random.choice(answers)
            
            return ''


使用时：

.. code::

    class ConversationSystem(object):
        
        patterns = [
            ('你好[吗啊呀]?', ['你好', '嗨']),
            ('你是谁', ['我是一个AI', '我是个机器人']),
            ('你(多大|几岁)[吗啊呀]?', ['我还年轻', '这是个秘密']),
            ('你在(哪里|哪儿|什么地方)', ['我在云端', '我在你的身边']),
            ('.*', ['你说什么', '不好意思，没明白你的话'])
        ]

        def __init__(self):
            self.chat_engine = RegexChatAgent(ConversationSystem.patterns)

        def reply(self, uid, message):
            return self.chat_engine.reply(message)
        
        def interact_cli(self):
            while True:
                query = input('User:')
                if query == 'Q':
                    break
                print('AI:', self.reply('UserA', query))

    conv_system = ConversationSystem()
    conv_system.interact_cli()



运行上面代码，我们可以得到：

::

    User: 你好
    AI: 嗨
    User: 你是谁
    AI: 我是个机器人
    User: 你多大了
    AI: 我还年轻
    User: 你在哪里呢
    AI: 我在你的身边
    User: Q


带模板的脚本引擎
""""""""""""""""

前面我们说过，一个脚本引擎通常上会支持从输入中提取内容，映射并处理回答模板后返回作为输出。

现在我们在基于正则的脚本引擎上支持这些功能。

.. code::

    import re
    import random

    class TemplateChatAgent(object):

        DEFAULT_MARK = '*'

        def __init__(self, patterns):
            self.patterns = [(re.compile(regex), templates) for regex, templates in patterns]
        
        def _process_match(self, match, templates):
            if not match:
                return ''
            
            candidate = ''
            named_groups = match.groupdict()
            if named_groups:
                for k, v in named_groups.items():
                    if v in templates:
                        candidates = templates[v]
                        candidate = random.choice(candidates)

                        break

            if not candidate and TemplateChatEngine.DEFAULT_MARK in templates:
                candidate = random.choice(templates[TemplateChatEngine.DEFAULT_MARK])
            
            if named_groups:
                for k, v in named_groups.items():
                    candidate = candidate.replace('{%s}' % k, v)
            
            return candidate
        
        def reply(self, message):
            for regex, templates in self.patterns:
                match = regex.match(message)
                if match:
                    return self._process_match(match, templates)
            
            return ''

.. code::

    class ConversationSystem(object):
        
        patterns = [
            (r'你好[吗啊呀]?', {'*': ['你好', '嗨']}),
            (r'你是男是女', {'*': ['你觉得呢', '你呢']}),
            (r'你是谁', {'*': ['我是一个AI', '我是个机器人']}),
            (r'你(多大|几岁)[吗啊呀]?', {'*': ['我还年轻', '这是个秘密']}),
            (r'你在(哪里|哪儿|什么地方)', {'*': ['我在云端，你在哪儿', '我在你的身边，你在哪里呢']}),
            (r'我是(?P<Gender>[男女])(的|生|人)', {'男': ['帅哥好'], '女': ['美女好']}),
            (r'我也?在(?P<Place>.+)', {'北京': ['北京现在很冷吧'], '*': ['你在{Place}？']}),
            (r'.*', {'*': ['你说什么', '不好意思，没明白你的话']})
        ]

        def __init__(self):
            self.chat_engine = TemplateChatAgent(ConversationSystem.patterns)

        def reply(self, uid, message):
            return self.chat_engine.reply(message)
        
        def interact_cli(self):
            while True:
                query = input('User:')
                if query == 'Q':
                    break
                print('AI:', self.reply('UserA', query))

    conv_system = ConversationSystem()
    conv_system.interact_cli()


::

    User: 你好
    AI: 你好
    User: 你是谁
    AI: 我是个机器人
    User: 你是男是女
    AI: 你觉得呢
    User: 男的
    AI: 不好意思，没明白你的话
    User: 你在哪里
    AI: 我在云端，你在哪儿
    User: 我也在云端
    AI: 你在云端？
    User: Q


如果我们把 `patterns` 中的规则写到一个模板文件中，并从模板文件中加载规则，这样一个脚本引擎就完成了。
它可以加载模板，模板的脚本语言就是由正则表达式组成的。

比如，我们可以使用如下的格式保存规则到文件中。

::

  - 你好[吗啊呀]?
    - *
      - 你好
      - 嗨
  - 你是男是女
    - *
      - 你觉得呢
      - 你呢
  - 我也?在(?P<Place>.+)
    - 北京
      - 北京现在很冷吧
    - *
      - 你在{Place}？


使用RiveScript作为脚本引擎
----------------------------

使用脚本引擎构建对话系统已有很长的历史，几种常见的脚本引擎：

* Eliza
* AIML
* RiveScript
* ChatScript
* SuperScript

这些脚本引擎功能大同小异，可以根据项目情况进行选择。如果你喜欢雍容华贵的XML，可能会选择AIML；如果你是Javascript的爱好者，可以会选择SupserScript。

下面我们使用 `RiveScript` 来构建一个对话系统。

RiveScript的脚本定义，以及编写规则在官方文档里写得很详细，这里不再赘述。

我们编写两个脚本文件，一个保存聊天机器人基本信息。

::

    ! version = 2.0

    > begin
        + request // This trigger is tested first.
        - {ok}    // An {ok} in the response means it's okay to get a real reply
    < begin

    // Bot Variables
    ! var name     = 小小
    ! var fullname = 小小
    ! var age      = 5
    ! var birthday = 10月12日
    ! var sex      = 男的
    ! var city     = 北京
    ! var color    = 蓝色

    // Set arrays
    ! array malenoun   = 男 男生 男的 男人 帅哥
    ! array femalenoun = 女 女生 女的 女人 美女


``myself.rive`` 这个脚本文件提供规则来回答与机器人自身相关的一些问题。

::

    // Tell the user stuff about ourself.

    + 小小
    - 在呢

    + 小小 *
    - 在呀 {@<star>}

    + (你是谁|你叫什么名字|你的名字是什么)
    - 我是 <bot name>。
    - 你可以叫我 <bot name>。

    + 你多大了
    - 我 <bot age> 岁了。
    - <bot age> 岁呀。

    + 你是(@malenoun)还?是(@femalenoun)
    - 我是 <bot sex>。

    + 你在(哪里|哪儿|什么地方)
    - 我在 <bot city>。

    + 你喜欢什么颜色
    - 当然是 <bot color>。


然后把这两个文件保存在一个 ``brain`` 文件夹下。

.. code-block ::

    class RiveChatAgent(object):
        
        def __init__(self, brain_path):
            self.rive = RiveScript(utf8=True)
            self.unicode_punctuation = re.compile(r'[。，！？；：.,!?;:]')
            self.rive.load_directory(brain_path)
            self.rive.sort_replies()
            
        def reply(self, message):
            response = self.rive.reply("", message)
            
            if response == '[ERR: No Reply Matched]':
                response = ''
            
            return response


最后在对话系统中使用这个基于 ``RiveScript`` 的对话引擎。

.. code::

    class ConversationSystem(object):

        def __init__(self):
            self.chat_engine = RiveChatAgent('brain')

        def reply(self, uid, message):
            return self.chat_engine.reply(message)
        
        def interact_cli(self):
            while True:
                query = input('User:')
                if query == 'Q':
                    break
                print('AI:', self.reply('UserA', query))

    conv_system = ConversationSystem()
    conv_system.interact_cli()


::

    User:你是谁
    AI: 我是 小小。
    User:你多大了
    AI: 我 5 岁了。
    User:你在哪里
    AI: 我在 北京。
    User:你是男是女
    AI: 我是 男的。


到目前为止，我们通过使用脚本引擎，实现了一个简单的对话系统。由于脚本引擎只能根据人工编写的脚本来进行对话，所以不能涵盖很多话题。接下来，我们来看一下如何利用检索的方式来产生回答进行对话。

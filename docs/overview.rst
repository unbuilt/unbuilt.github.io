概述
====

创造一个跟人类自己一样聪明的机器人，是人们一直有的梦想。`和机器人一起进化` [Robot]_ 这本书是讨论了很多各种各样的机器人，科幻电影中的，现实世界中的，以及未来憧憬中的。

对话是智能机器人应该有的基本功能，对话系统也可以脱离实体机器人而存在。这里的对话系统，也可以称为聊天机器人。
`小冰`_ 和 `Replika`_ 是两个比较有意思的以对话为核心功能的人工智能产品。

`From Eliza to XiaoIce: Challenges and Opportunities with Social Chatbots` [Xiaoice]_ 
这篇文章介绍了聊天机器人的发展历史，并以小冰为例，讨论了构建聊天机器人的对话、视觉及技能等重要技术。

`The Design and Implementation of XiaoIce, an Empathetic Social Chatbot` [Xiaoice2]_ 详细描述了小冰的设计与实现。

`A Survey on Dialogue Systems: Recent Advances and New Frontiers` [Survey]_ 将对话系统划分成了面向任务和非面向任务两种类型，并分别介绍了研究进展及一些研究方向。

`Open-Domain Conversational Agents: Current Progress, Open Problems, and Future Directions` [FAIR]_ 也总结了对话系统的进展，存在的问题，以及未来的方向。

`Building A User-Centric and Content-Driven Socialbot` [SoundingBoard]_ 这篇博士论文介绍了如何构建一个以用户为中心以内容驱动的对话系统。


这里，我们从最简单的代码开始介绍实现一个对话系统的基本方法。


对话系统
--------------

对话系统的功能是与用户进行对话，基本形式是接受一句用户输入的话，输出一句回答的话。会话系统通常会记录并利用对话历史以产生出更合理的回答。

一个最简单的对话系统可以用下面的代码实现，它使用用户的输入作为对话系统的输出：

.. code-block:: python

    class ConversationSystem(object):

        def __init__(self):
            pass

        def reply(self, uid, message):
            return message
        
        def interact_cli(self):
            while True:
                query = input('User:')
                if query == 'Q':
                    break
                print('AI:', self.reply('User', query))


如果我们运行这个对话系统：

.. code-block:: python

    conv_system = ConversationSystem()
    conv_system.interact_cli()

会产生如下输出：

::

    User: 你好
    AI: 你好
    User: 很高兴认识你
    AI: 很高兴认识你
    User: 再见
    AI: 再见
    User: Q

上面的对话系统只会重复用户的输入，其实产生输出的方式会有很多种。我们这里先把针对一句用户的输入产生一句相应的输出的这一基本功能抽象为 ``ChatAgent``：

.. code-block:: python

    class ChatAgent(object):

        def reply(self, message):
            pass


并实现一个 ``EchoChatAgent`` 来改写前面的对话系统。

.. code-block:: python

    class EchoChatAgent(object):

        def reply(self, message):
            return message

    class ConversationSystem(object):

        def __init__(self):
            self.chat_engine = EchoChatAgent()

        def reply(self, uid, message):
            return self.chat_engine.reply(message)
        
        def interact_cli(self):
            while True:
                query = input('User:')
                if query == 'Q':
                    break
                print('AI:', self.reply('User', query))


再次通过命令行与这个对话系统进行交互，我们可以得到与上面相同的结果。

上面的对话系统只会重复用户的输入，如何产生其他的回复呢？通常有以下三种方法：

* :doc:`script`
* :doc:`retrieval`
* :doc:`generative`



.. _小冰: http://xiaoice.ai/
.. _Replika: https://replika.ai

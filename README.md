# NMT zh2en


## 依赖

- Python 3.5
- PyTorch 0.4

## dataset
[https://www.statmt.org/](https://www.statmt.org/)

## 

### Preperation
```bash
$ extract.py
$ python multi_threads_pre_process.py
```

### Trainning
```bash
$ python train.py
```


### translator

```bash
$ python translator.py
```

<pre>
请输入中文: (输入q退出)
我爱你
译文 i love you
请输入中文: (输入q退出)
我很爱你
译文 i love you very much
请输入中文: (输入q退出)
国王死了
译文 the king is dead
请输入中文: (输入q退出)
一起旅行
译文 together traveling
请输入中文: (输入q退出)
跟她谈谈
译文 talk to her
请输入中文: (输入q退出)
你还打算重新开始游戏吗
译文 are you gon na start over again ?
请输入中文: (输入q退出)
什么都没有改变不用担心
译文 nothing has changed
请输入中文: (输入q退出)
一点作用都没起
译文 and do n t work
请输入中文: (输入q退出)
一点儿也不在乎
译文 do n t even care about
请输入中文: (输入q退出)
q
</pre>
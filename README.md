# GPT

### 使用指南

#### 1. 存储位置与执行
模型存放在models文件夹中，在这个文件夹中，/models/model.py是存放整个模型的地方
训练使用的是/open_instruct/finetune.py文件
然后这个是使用/scripts/finetune_with_accelerate.sh进行调用
调用的参数可以在/scripts/finetune_with_accelerate.sh和/ds_configs/stage2.conf中进行设置
/conversation.py是一个用于进行测试正常对话的文件，可以在文件中
![Alt text](/images/image.png)
修改输入，然后直接在命令行运行python conversation.py即可

### 2. 训练配置
想要执行训练，直接在命令行运行./scripts/finetune_with_accelerate.sh即可
在/scripts/finetune_with_accelerate.sh中可以修改一些训练的参数
因为模型都是已经确定的，所以就没有将模型的路径写进去
可以在--output_dir修改模型保存的地点

### 3. 加载与使用模型
首先在引入models.model里面的GPT类之后，便可以直接生成一个模型
然后使用load_model进行模型参数的读取（注意这里是读取.bin文件）
然后再将调用模型的to_device('cuda')即可
![Alt text](/images/image-2.png)
由于模型的输入是经过xlmr模型tokenize之后的结果，所以我们可以进行如下操作：
1. 将str类型的text和answer按照下面的格式放到一个sample中，如果是生成文字，answer可以不要，也就是说completion可以直接赋值为None
```
sample = {
    'prompt': [text1, text2],
    'completion': [answer1, answer2]
}
```
2. 调用/conversation.py中的solve_data将当前sample进行转换，具体可以见下图或者见/conersation.py中
![Alt text](/images/image-4.png)

### 4. 从json文件中加载数据集
这个参见/open-instructs/finetune.py中的
```
read_data()
solve_data()
get_dataloader()
CustomDataset
```
三个方法和CustomDataset类
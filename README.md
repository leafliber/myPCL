# myPCL

## Training:
#### 参数解析
| 简写参数 |      全称参数       | 描述                                     |
|:----:|:---------------:|:---------------------------------------|
|  -a  |     --arch      | 指定主干网络类型，如：resnet-18，resnet-50         |
|  -j  |    --workers    | 指定线程数，默认为4                             |
|      |    --epochs     | 总训练循环次数，默认为200                         |
|      | --warmup-epoch  | 有监督epoch次数，默认为100                      |
|      |    --exp-dir    | 输出路径，默认为experiment                     |
|  -b  |  --batch-size   | 一批的数量，默认为8，必须为标签数的倍数                   |
| -lr  | --learning-rate | 学习率，默认为0.03                            |
|      |      --cos      | 使用cosine学习率                            |
|      |   --schedule    | 指定学习率下降的epoch，默认为[120,160]，只在cos未指定时生效 |
|      |   --momentum    | 优化器的动量餐宿，默认为0.9                        |
| --wd | --weight-decay  | SSPCL模型的权重衰减，默认为1e-4                   |
|      |    --low-dim    | 输出维度，默认为128                            |
|      |  --num-cluster  | 聚类个数，默认为'20,25,30'                     |
|      |     --pcl-r     | 负例对，需要小于聚类个数，默认为16                     |
|      |    --moco-m     | SSPCL中ME更新参数使用的动量，默认为0.999             |
|      |      --mlp      | 设置即为使用mlp，无参数，参考PCL模型                  |
|      |  --temperature  | softmax层温度参数，默认为0.2                    |
|  -p  |  --print-freq   | 显示频率，默认为每10个数据                         |
|      |   --save-freq   | 保存模型的频率，默认为每10个epoch                   |
|      |  --world-size   | 总训练程序数量，默认为1                           |
|      |     --rank      | 此训练程序编号，默认编号0                          |
|      |   --dist-url    | 多程序训练连接地址，此参数参照pytorch分布式训练解释          |
|      | --dist-backend  | 默认为nccl                                |
|      |      --gpu      | 用于训练的gpu编号                             |
|      |     --seed      | 随机数种子，默认为自动生成                          |
|      |    --resume     | 需要载入的模型位置                              |
|      |  --start-epoch  | 训练起始的epoch，与resume配合使用                 |
|      |                 |                                        |

#### 用例
<pre>
python main.py -a resnet18 --lr 0.03 --batch-size 8 --workers 4 --temperature 0.2 --mlp --aug-plus --cos --dist-url "tcp://localhost:10001" --world-size 1 --rank 0 --warmup-epoch 100 --epochs 100 --exp-dir exp images
</pre>

## Testing:
#### 参数解析
**如训练时修改了以上默认的参数，在测试时也需要指定**
以下是必须要设置的参数

| 简写参数 |     全称参数     | 描述        |
|:----:|:------------:|:----------|
|      | --pretrained | 需要载入模型的路径 |

#### 用例
<pre>
python test_svm.py --pretrained exp/checkpoint_0199.pth.tar
</pre>

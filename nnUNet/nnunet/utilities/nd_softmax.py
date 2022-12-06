#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F

# 简易函数，
# 比如a = lambda x:x+10
# a(10)  输出20
# 这里的耶斯就是,softmax_helper 就是f.softmax()
# dim=1 输入的x是 bcxyz ,所以是在channel上进行softmax
softmax_helper = lambda x: F.softmax(x, 1)


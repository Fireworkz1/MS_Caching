# MS_Caching

### 节点和边的初始化：main.py

### 训练参数：train.py

#### state维数：135

六个SN，十个EN，十九条边，五种微服务

六个SN和十个EN每个有七个状态位，分别为当前剩余内存，CPU和五个微服务的部署情况（Onehot编码）

十九条边各有一个状态位

请求的微服务的参数为三个状态位，
请求的微服务编号为一个状态位

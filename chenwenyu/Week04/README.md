
[TOC]

### Project Organization
bert_tune_proj/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 主文件
│   ├── bert_model.py    # 训练代码
│   ├── predictor.py     # 预测服务类
│   ├── schemas.py       # Pydantic模型
│   └── models/          # 保存训练好的模型
├── assets
│   ├──models/           #google-bert-chinese model
│   ├──waimai_10k.csv     #数据集
│   └──post_data.json     #AB测试用内容
├── README.md             #使用说明，存在的问题等
└── train.py            # 训练入口脚本 

### start train

cd ~/work/bert_tune_proj
python train.py
......
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at ./assets/models/google-bert/bert-base-chinese and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Using device: cpu
------------Epoch: 0 ----------------
Epoch: 0, Average training loss: 0.7372
Accuracy: 1.0000
Average testing loss: 0.0956
-------------------------------
Model saved to ./app/models/bert-finetuned-epoch0
🎉 新的最佳模型保存，准确率: 1.0000
------------Epoch: 1 ----------------
Epoch: 1, Average training loss: 0.0380
Accuracy: 1.0000
Average testing loss: 0.0061
-------------------------------
------------Epoch: 2 ----------------
Epoch: 2, Average training loss: 0.0037
Accuracy: 1.0000
Average testing loss: 0.0013
-------------------------------
------------Epoch: 3 ----------------
Epoch: 3, Average training loss: 0.0012
Accuracy: 1.0000
Average testing loss: 0.0007
-------------------------------

训练完成！最佳准确率: 1.0000

### run service
cd ~/work/bert_tune_proj
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
[1] 21599
(pytorch_d2l) 192:bert_tune_proj wenyuc$ INFO:     Will watch for changes in these directories: ['/Users/wenyuc/work/bert_tune_proj']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [21599] using StatReload
INFO:     Started server process [21602]
INFO:     Waiting for application startup.
尝试加载模型从: app/models/bert-finetuned-epoch0
路径是否存在: True
INFO:app.predictor:Using device: cpu
INFO:app.predictor:模型目录文件: ['model.safetensors', 'label_encoder.pkl', 'tokenizer_config.json', 'special_tokens_map.json', 'config.json', 'vocab.txt']
INFO:app.predictor:开始加载模型...
INFO:app.predictor:模型加载成功
INFO:app.predictor:开始加载分词器...
INFO:app.predictor:分词器加载成功
INFO:app.predictor:标签编码器加载成功，类别: [0, 1]
✅ Model loaded successfully
Predictor type: <class 'app.predictor.BertPredictor'>
Predictor device: cpu
INFO:     Application startup complete.
#### Tips 
查找占用8000端口的进程
lsof -i :8000
或者使用
netstat -anv | grep 8000
### Stop Service
pkill -f uvicorn

### test api
#### 1. 健康检查
curl http://localhost:8000/health
Health check - predictor: <app.predictor.BertPredictor object at 0x12c06b220>
INFO:     127.0.0.1:56039 - "GET /health HTTP/1.1" 200 OK
{"status":"healthy","model_loaded":true,"device":"cpu"}

#### 2. Debug predict return value
curl http://localhost:8000/debug-predict
INFO:     127.0.0.1:56101 - "GET /debug-predict HTTP/1.1" 200 OK
{"type":"<class 'list'>","value":"[{'text': '这', 'predicted_label': 1, 'predicted_class': 1, 'confidence': 0.900057315826416}]","is_numpy":false,"converted":[{"text":"这","predicted_label":1,"predicted_class":1,"confidence":0.900057315826416}],"converted_type":"<class 'list'>"}

#### 2. 单个预测
curl -X POST "http://localhost:8000/predict/这个产品很好用"
INFO:     127.0.0.1:56169 - "POST /predict/%E8%BF%99%E4%B8%AA%E4%BA%A7%E5%93%81%E5%BE%88%E5%A5%BD%E7%94%A8 HTTP/1.1" 200 OK
{"text":"这个产品很好用","predicted_label":"1","predicted_class":1,"confidence":0.9001}

#### 3. 批量预测(还在调试阶段)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["这个很好", "那个不好", "一般般"]}'

curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"texts": ["这个很好", "那个不好"]}'

### AB测试结果

ab -n 100 -c 1 -p ./assets/post_data.json -T "application/json" http://127.0.0.1:8000/predict
(pytorch_d2l) 192:bert_tune_proj wenyuc$ ab -n 100 -c 1 -p ./assets/post_data.json -T "application/json" http://127.0.0.1:8000/predict
This is ApacheBench, Version 2.3 <$Revision: 1913912 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 127.0.0.1 (be patient)...INFO:     127.0.0.1:51669 - "POST /predict HTTP/1.0" 200 OK
INFO:     127.0.0.1:51670 - "POST /predict HTTP/1.0" 200 OK
......
..done


Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /predict
Document Length:        93 bytes

Concurrency Level:      1
Time taken for tests:   3.813 seconds
Complete requests:      100
Failed requests:        0
Total transferred:      23700 bytes
Total body sent:        17500
HTML transferred:       9300 bytes
Requests per second:    26.23 [#/sec] (mean)
Time per request:       38.130 [ms] (mean)
Time per request:       38.130 [ms] (mean, across all concurrent requests)
Transfer rate:          6.07 [Kbytes/sec] received
                        4.48 kb/s sent
                        10.55 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.0      0       0
Processing:    34   38  10.3     36     136
Waiting:       34   38  10.3     36     135
Total:         34   38  10.3     36     136

Percentage of the requests served within a certain time (ms)
  50%     36
  66%     37
  75%     37
  80%     38
  90%     40
  95%     44
  98%     57
  99%    136
 100%    136 (longest request)

$ab -n 100 -c 5 -p ./assets/post_data.json -T "application/json" http://127.0.0.1:8000/predict
(pytorch_d2l) 192:bert_tune_proj wenyuc$ ab -n 100 -c 5 -p ./assets/post_data.json -T "application/json" http://127.0.0.1:8000/predict
This is ApacheBench, Version 2.3 <$Revision: 1913912 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 127.0.0.1 (be patient)...INFO:     127.0.0.1:52001 - "POST /predict HTTP/1.0" 200 OK
INFO:     127.0.0.1:52002 - "POST /predict HTTP/1.0" 200 OK
......
..done


Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /predict
Document Length:        93 bytes

Concurrency Level:      5
Time taken for tests:   3.711 seconds
Complete requests:      100
Failed requests:        0
Total transferred:      23700 bytes
Total body sent:        17500
HTML transferred:       9300 bytes
Requests per second:    26.95 [#/sec] (mean)
Time per request:       185.545 [ms] (mean)
Time per request:       37.109 [ms] (mean, across all concurrent requests)
Transfer rate:          6.24 [Kbytes/sec] received
                        4.61 kb/s sent
                        10.84 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.1      0       0
Processing:    55  180  42.8    176     274
Waiting:       34  119  53.6    128     273
Total:         55  180  42.8    176     274

Percentage of the requests served within a certain time (ms)
  50%    176
  66%    183
  75%    198
  80%    212
  90%    244
  95%    274
  98%    274
  99%    274
 100%    274 (longest request)

(pytorch_d2l) 192:bert_tune_proj wenyuc$ ab -n 100 -c 10 -p ./assets/post_data.json -T "application/json" http://127.0.0.1:8000/predict
This is ApacheBench, Version 2.3 <$Revision: 1913912 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 127.0.0.1 (be patient)...INFO:     127.0.0.1:52306 - "POST /predict HTTP/1.0" 200 OK
......
..done


Server Software:        uvicorn
Server Hostname:        127.0.0.1
Server Port:            8000

Document Path:          /predict
Document Length:        93 bytes

Concurrency Level:      10
Time taken for tests:   3.680 seconds
Complete requests:      100
Failed requests:        0
Total transferred:      23700 bytes
Total body sent:        17500
HTML transferred:       9300 bytes
Requests per second:    27.17 [#/sec] (mean)
Time per request:       368.042 [ms] (mean)
Time per request:       36.804 [ms] (mean, across all concurrent requests)
Transfer rate:          6.29 [Kbytes/sec] received
                        4.64 kb/s sent
                        10.93 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.1      0       0
Processing:    52  360  42.9    353     446
Waiting:       48  228 104.5    222     445
Total:         52  360  42.9    354     446

Percentage of the requests served within a certain time (ms)
  50%    354
  66%    354
  75%    363
  80%    375
  90%    444
  95%    446
  98%    446
  99%    446
 100%    446 (longest request)

 ### 存在的问题
 1. 批量预测还在继续调试

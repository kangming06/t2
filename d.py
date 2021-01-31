from imageai.Prediction import ImagePrediction
import os
import time
import requests
start = time.time()
 
#当前路径 包含需要预测的图片，模型文件
execution_path = os.getcwd()
 
#创建预测类
prediction = ImagePrediction()
 
#设置预测模型 有以下四种
#SqueezeNet
#prediction.setModelTypeAsSqueezeNet()
#prediction.setModelPath(os.path.join(execution_path, "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))
 
#ResNet50
#prediction.setModelTypeAsInceptionV3()
#prediction.setModelPath(os.path.join(execution_path, "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
 
#InceptionV3
#prediction.setModelTypeAsResNet()
#prediction.setModelPath(os.path.join(execution_path, "resnet50.h5"))
 
#DenseNet121
prediction.setModelTypeAsDenseNet()
prediction.setModelPath(os.path.join(execution_path, "DenseNet-BC-121-32.h5"))
 
prediction.loadModel()
 
#预测图片，以及结果预测输出数目
predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "1.jpg"), result_count=5 )
 
#结束计时
end = time.time()
 
#输出结果
#输出结果
outlist={}
outnumber=0
zh_ch=0
for eachPrediction, eachProbability in zip(predictions, probabilities):
	if eachProbability>=0.6:
		r = requests.get("http://fanyi.youdao.com/translate",params={'doctype': 'json','type': 'AUTO','i':eachPrediction})
		result = r.json()
		zh_ch = result['translateResult'][0][0]["tgt"]
		print(translate_result)
		outlist[outnumber]=zh_ch
		outnumber+=1
	print(outlist)


print ("\ncost time:",end-start)

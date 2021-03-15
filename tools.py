import os
import csv
import cv2
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.lite.python import interpreter as interpreter_wrapper
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

os.environ["CUDA_VISIBLE_DEVICES"] = "" # do not occupy GPU

#划分数据集
def divide_dataset(data_dir,num_test):
    """
    data_dir：目录如下
    |-train
    |-----depth
    |-----rgb
    |-val(不参与分割数据集，用于测试)
    |-train.csv
    |-val.csv

    num_test：指定测试集大小
    """
    train_csv = open(os.path.join(data_dir,"train.csv"),"w",encoding='utf-8')
    val_csv = open(os.path.join(data_dir,"val.csv"),"w",encoding='utf-8')
    train_writer = csv.writer(train_csv)
    val_writer = csv.writer(val_csv)

    image_list = os.listdir(os.path.join(data_dir,"train/rgb"))
    num_list = [x.split('.')[0] for x in image_list]
    random.shuffle(num_list)

    val_list = num_list[:num_test]
    train_list = num_list[num_test:]

    for val in val_list:
        val_writer.writerow([f'train/rgb/{val}.png',f'train/depth/{val}.png'])

    for train in train_list:
        train_writer.writerow([f'train/rgb/{train}.png',f'train/depth/{train}.png'])

    print(f"生成验证集数量{len(val_list)}，测试集数量{len(train_list)}")
    train_csv.close()
    val_csv.close()

#计算FLOPs
def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops

#转换为TFlite模型
def convert_to_tflite(saved_model_dir):
    def convert_from_save_model(model_save_path = saved_model_dir, type = 'normal'):
        converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)

        assert type in ['normal','float16','int8',"only-opt"]
        if type == "normal":
            tflite_save_path = os.path.join(model_save_path, 'tflite_model.tflite')
        elif type == "only-opt":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_save_path = os.path.join(model_save_path, 'tflite_model_opt.tflite')
        elif type == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_save_path = os.path.join(model_save_path, 'tflite_model_float16.tflite')
            '''
        elif type == "int8": # can't use int8 in this competiton
            #converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.representative_dataset = representative_dataset
            # Ensure that if any ops can't be quantized, the converter throws an error
            # Set the input and output tensors to uint8 (APIs added in r2.3)
            converter.inference_input_type = tf.uint16
            converter.inference_output_type = tf.uint16
            tflite_save_path = os.path.join(model_save_path, 'tflite_model_int8.tflite')
            '''
        tflite_model = converter.convert()
        if tflite_save_path != None:
            with open(tflite_save_path, 'wb') as f:
                f.write(tflite_model)
                print(f"已生成模型：{tflite_save_path}")

    types = ['normal','float16','only-opt']
    for type in types:
        convert_from_save_model(saved_model_dir,type = type)

#生成可提交图片
def generate_results(tflite_model, val_dir, scale_value, normal_val,results_dir=''):
    '''
    :param tflite_model: (必填)tflite模型地址
    :param val_dir:（必填）验证集路径
    :param results_dir: （选填）目标生成文件的路径，若不赋值默认为读取模型位置
    :param scale_value: （必填）图片下采样率
    '''
    interpreter = interpreter_wrapper.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if not results_dir:
        results_dir = os.path.join(os.path.dirname(tflite_model), 'result')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    img_list = os.listdir(val_dir)
    assert len(img_list) == 500,f'验证集图片数量为{len(img_list)}，原始验证集数量为500'

    for img in tqdm(img_list):
        #读取图片
        img_path = os.path.join(val_dir,img)

        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=3, dtype=tf.dtypes.uint8)
        image = np.asarray(image).astype(np.float32)
        #图片预处理
        image_shape = np.shape(image)
        image /= 255.0

        image_new = cv2.resize(image,
                               (int(image_shape[1] / scale_value), int(image_shape[0] / scale_value)),
                               interpolation=cv2.INTER_LINEAR)
        image_new = np.reshape(image_new, (1, int(image_shape[0] / scale_value), int(image_shape[1] / scale_value), 3))

        #推理
        interpreter.set_tensor(input_details[0]['index'], image_new)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)
        results *= normal_val
        results = results.astype(np.uint16)

        result_path = os.path.join(results_dir, img)
        cv2.imwrite(result_path, results)
    print(f'结果生成从存放在{results_dir}')

def anaylse(label_dir,is_single_img):
    img_list = os.listdir(label_dir)
    non_zero = 0
    min,max = 10000,0
    print("分析数据集中... ...")
    if not is_single_img:
        for img in tqdm(img_list):
            image = tf.io.read_file(os.path.join(label_dir,img))
            image = tf.image.decode_png(image,channels=1,dtype=tf.dtypes.uint16)
            image = image.numpy()
            if np.min(image) == 0:
                image = np.where(image==0,10000,image)
            else:
                non_zero += 1
            max = np.max(image) if max<np.max(image) and np.max(image) not in [39999,39998,39997] else max
            min = np.min(image) if min>np.max(image) and np.min(image) not in [682,786,1146,809] else min

        print(f"拥有非零图片的数量为：{non_zero},图片数值范围:{min}~{max}")
    else:
        image = tf.io.read_file(os.path.join(label_dir,img_list[0]))
        image = tf.image.decode_png(image, channels=1, dtype=tf.dtypes.uint16)
        image = image.numpy()
        image = np.reshape(image,(480*640))
        image = np.sort(image)

        distribute = np.unique(image)

        print(distribute)
        print(np.shape(distribute))

def calculate_mean_std(dataset_dir,is_label, is_image):
    image_dir = os.path.join(dataset_dir,'rgb')
    label_dir = os.path.join(dataset_dir,'depth')
    image_list = os.listdir(image_dir)
    label_list = os.listdir(label_dir)

    if is_label:
        print("计算标签中... ...")
        label_mean = []
        label_std = []
        for label in tqdm(label_list):
            label_path = os.path.join(label_dir,label)
            label = tf.io.read_file(label_path)
            label = tf.image.decode_png(label,channels=1,dtype=tf.dtypes.uint16)
            label = label.numpy()
            label_mean.append(np.mean(label))
            label_std.append(np.std(label))
        label_mean = np.array(label_mean)
        label_std = np.array(label_std)
        print(f"标签的均值为：{np.mean(label_mean)},标准差为：{np.mean(label_std)}")

    if is_image:
        image_mean,image_std = [],[]
        for image in tqdm(image_list):
            img_path = os.path.join(image_dir,image)
            image = cv2.imread(img_path)
            mean,std = cv2.meanStdDev(image)
            image_mean.append(mean)
            image_std.append(std)
        image_mean_array = np.array(image_mean)
        image_std_array = np.array(image_std)
        image_mean = image_mean_array.mean(axis=0, keepdims=True)
        image_std = image_std_array.mean(axis=0, keepdims=True)
        print(f"图片的均值为：{image_mean[0][::-1]},标准差为：{image_std[0][::-1]}")

    pass


if __name__ == "__main__":

    #divide_dataset(data_dir = '/home/share/competition/depth_estimation/',num_test = 100)

    #saved_model_dir = '/home/wupeilin/project/depth_estimation/models/1615520288-n227-e100-bs32-lr0.002-rfnet_Norm1/40_0.52090/'
    #convert_and_test(saved_model_dir = saved_model_dir)

    #tflite_model = '/home/wupeilin/project/depth_estimation/models/1615520288-n227-e100-bs32-lr0.002-rfnet_Norm1/40_0.52090/tflite_model_float16.tflite'
    #val_dir = '/home/share/competition/depth_estimation/val/'
    #generate_results(tflite_model=tflite_model,val_dir=val_dir,scale_value=4,normal_val=1600)

    #label_dir = '/home/share/competition/depth_estimation/train/depth'
    #naylse(label_dir,is_single_img=True)

    #calculate_mean_std(dataset_dir='/home/share/competition/depth_estimation/train/',is_image=True,is_label=Falsefu)

    pass

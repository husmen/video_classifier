import os
import time
import resource
import cv2
import numpy as np
from threading import Thread
from skimage.measure import compare_ssim as ssim

tags = {"0001.mp4":"S","0002.mp4":"N","0003.mp4":"S","0004.avi":"N","0005.mp4":"S","0006.mpg":"N","0007.mp4":"S","0008.mp4":"N","0009.mp4":"S","0010.mp4":"N","0011.mp4":"S","0012.mp4":"S","0013.mp4":"S","0014.mp4":"N","0015.mpg":"S","0016.avi":"S","0017.avi":"S", "0018.mp4":"N", "0019.mp4":"N", "0020.mp4":"N"}
dataset_path = "/home/husmen/Workspace/PARDUS/dosyalar_/mp4-avi-mpg"
media = []
results = []
times = []

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(imageA, imageB, algo="ssim"):
    if algo == "ssim":
        m = ssim(imageA, imageB)  # Better results -- Structural Similarity
    else:
        m = mse(imageA, imageB)  # Faster results -- Mean squared error
    return m

# Calculate accuracy, Precision, Recall and F1 Score
def score(TP, TN, FP, FN):
    Acc = (TP + TN) / (TP + TN + FP + FN)
    Pre = TP / (TP + FP)
    Rec = TP / (TP + FN)
    F1 = 2 * (Rec * Pre) / (Rec + Pre)
    return Acc, Pre, Rec, F1

if __name__=="__main__":
    """ main routine """
    for _ in os.listdir(dataset_path+'/'):
        media.append(dataset_path+'/'+_)

    method = "mse" # choose method: "mse" or "ssim"
    if method == "ssim":
        threshold = 0.76
    elif method == "mse":
        threshold = 1000

    counter = 0
    startTime2 = int(round(time.time()))
    for videoPath in media:
        startTime = int(round(time.time()))
        dim = (192, 144)
        resize_flag = False
        vid = videoPath.split("/")[-1]
        print("### {}/{} Processing {} ###".format(counter, len(media), vid))
        video = cv2.VideoCapture(videoPath)
        fps = video.get(cv2.CAP_PROP_FPS)
        res = (video.get(cv2.CAP_PROP_FRAME_WIDTH),video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if res[1] > 144:
            r = res[0]/res[1]
            dim = (int(144 * r), 144)
            resize_flag = True
        counter += 1
        
        suc, img = video.read()
        count_1 = 0
        count_2 = 1
        img_processed = []
        while suc:
            if count_1%int(fps) == 0:
                img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if resize_flag:       
                    img_g = cv2.resize(img_g, dim)
                img_processed.append(img_g)
                count_2 += 1
            count_1 += 1
            suc, img = video.read()

        print("\tFPS: {} # Resolution: {} #: New Resolution: {} # Numer of frames: {}".format(round(fps,2), res, dim, len(img_processed)))

        x = 3 # Number of following seconds to compare to
        ratio = []
        points = [i for i in range(0,len(img_processed), int(len(img_processed)/4))[:4]]
        points.append(len(img_processed))

        def procces (start,end,count):
            for i in range(start, end):
                img1 = img_processed[i]
                for j in range(count):
                    i_2 = i + j
                    if i_2 < len(img_processed):
                        img2 = img_processed[i+j]
                        tmp = compare_images(img1, img2, algo = method)
                        ratio.append(tmp)

        threads = []
        threads.append(Thread(target=procces, args=(points[0], points[1], x)))
        threads.append(Thread(target=procces, args=(points[1], points[2], x)))
        threads.append(Thread(target=procces, args=(points[2], points[3], x)))
        threads.append(Thread(target=procces, args=(points[3], points[4], x)))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        avg = sum(ratio) / float(len(ratio))
        avg = round(avg,2)

        if method == "ssim" and avg >= threshold:
                results.append(("S",avg))              
        elif method == "mse" and avg <= threshold:
                results.append(("S",avg))
        else:
                results.append(("N",avg))
        endTime = int(round(time.time()))
        times.append(endTime-startTime)

    endTime2 = int(round(time.time()))
    time_delta = endTime2 - startTime2
    RAM_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print("\n\n### Results ###\n")
    #correct_counter = 0

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i, videoPath in enumerate(media):
        vid = videoPath.split("/")[-1]
        if results[i][0] == tags[vid]:
            if results[i][0] == "S":
                TP += 1
            else:
                TN +=1
            #correct_counter += 1
        else:
            if results[i][0] == "S":
                FP += 1
            else:
                FN += 1
        print("# {} # Result: {} ({}) # Actual: {} # Time: {}".format(vid, results[i][0], results[i][1], tags[vid], times[i]))
    Acc, Pre, Rec, F1 = score(TP, TN, FP, FN)

    #print("correct answer rate: {}/{}".format(correct_counter, len(media)))
    print("Total running time: {}".format(time_delta))
    print("Maximum RAM Usage: {} MB".format(RAM_max))
    print("Accuracy: {}\tPrecision: {}\tRecall: {}\tF1 Score: {}".format(Acc, Pre, Rec, F1))


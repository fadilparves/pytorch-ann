# Built ANN using Pytorch

TL;DR Built ANN using pytorch to predict if it is going to rain tomorrow or not going to rain tomorrow

### Data

Provided in the repo, it is AUS daily wheather data with target label know as `RainTomorrow`

### Some things to note
1. The network.gz.pdf is visualization of how the network looks
2. Only 2 hidden layers are being utilized
3. Output is applied with sigmoid function in order to return 1 or 0

### Performance
```
              precision    recall  f1-score   support

     No rain       0.85      0.96      0.90     19413
     Raining       0.74      0.40      0.52      5525

    accuracy                           0.84     24938
   macro avg       0.80      0.68      0.71     24938
weighted avg       0.83      0.84      0.82     24938
```
```
epoch 0
        Train set - loss: 2.513, accuracy: 0.779
        Test  set - loss: 2.517, accuracy: 0.778
        
epoch 100
        Train set - loss: 0.457, accuracy: 0.792
        Test  set - loss: 0.458, accuracy: 0.793
        
epoch 200
        Train set - loss: 0.435, accuracy: 0.801
        Test  set - loss: 0.436, accuracy: 0.8
        
epoch 300
        Train set - loss: 0.421, accuracy: 0.814
        Test  set - loss: 0.421, accuracy: 0.815
        
epoch 400
        Train set - loss: 0.412, accuracy: 0.826
        Test  set - loss: 0.413, accuracy: 0.827
        
epoch 500
        Train set - loss: 0.408, accuracy: 0.831
        Test  set - loss: 0.408, accuracy: 0.832
        
epoch 600
        Train set - loss: 0.406, accuracy: 0.833
        Test  set - loss: 0.406, accuracy: 0.835
        
epoch 700
        Train set - loss: 0.405, accuracy: 0.834
        Test  set - loss: 0.405, accuracy: 0.835
        
epoch 800
        Train set - loss: 0.404, accuracy: 0.834
        Test  set - loss: 0.404, accuracy: 0.835
        
epoch 900
        Train set - loss: 0.404, accuracy: 0.834
        Test  set - loss: 0.404, accuracy: 0.836
```

### Issues
1. Prediction for rain is not good for recall as the number of data sample for rain is only 20% and not rain has 80% (bias factor)
2. Current data preprocessing only able to push accuracy up to 84%, even with more epochs

### How to run
1. Clone this repo
2. Make sure you have [Pytorch](https://pytorch.org/get-started/locally/) installed
3. Run `main.py`
4. Optimize and have fun

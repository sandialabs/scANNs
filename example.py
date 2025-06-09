import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar100
from constraint import MinMax
from scann import sample_model, shannon_entropy

num_classes = 100
num_epochs = 50
num_samples = 40 

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train/255.0
    x_test = x_test/255.0
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    base = keras.applications.ResNet50(include_top=False, 
                                        weights=None,
                                        classes = num_classes,
                                        pooling = 'avg', 
                                        )
    
    initializer=keras.initializers.RandomUniform(minval=.25, maxval=.75, seed=123)
    
    output_layer = keras.layers.Dense(num_classes, 
                                activation='softmax',
                                kernel_initializer=initializer,
                                kernel_constraint=MinMax([-1.0, 1.0]))
    
    model = keras.models.Sequential([base, output_layer])

    model.summary()
    print("=======Training Model=======")

    model.build(input_shape=(None,32, 32, 3))
    model.summary()
    model.compile('adam', 'categorical_crossentropy', metrics='accuracy')
    model.fit(x_train,
              y_train, 
              batch_size=128,
              validation_data=(x_test, y_test),
              epochs = num_epochs)
    
    print("=======Non-Sampling Performance=======")
    result = model.predict(x_test, verbose=0)
    correct = np.sum(np.argmax(result, axis=1)==np.argmax(y_test, axis=1))
    accuracy = correct/len(y_test)
    print(f"Acc: {accuracy}")
    sampling_results_sum = np.zeros_like(result)
    print("=======Sampling Performance=======")
    #These predictions can be evaluated for second-place choice votes etc.
    original_weights = model.get_weights()
    for i in range(num_samples):
        model.set_weights(original_weights)
         #Note sample_model changes the model, 
         # returning a new model is just for convenience
        sampled_model = sample_model(model) 
        sampled_result = sampled_model.predict(x_test, verbose=0)
        sampling_results_sum += sampled_result
        if i == 0:
            sampling_results_history = [np.argmax(sampled_result, axis=1)]
        else:
            sampling_results_history = np.append(sampling_results_history, [np.argmax(sampled_result, axis=1)], axis=0)
        if (i+1)%10 == 0 or i == 0:
            accuracy = np.sum(np.argmax(sampling_results_sum, axis=1)==np.argmax(y_test, axis=1))/len(y_test)
            print(f"Afer {i+1} sample(s) - Acc: {accuracy}")


    print("=======Calculating Entropy Matrices=======")
    r=np.transpose(sampling_results_history)
    correct_H_ct = 0
    incorrect_H_ct = 0

    for ct, y in enumerate(y_test):
        H=shannon_entropy(np.bincount(r[ct], minlength=10)) # Entropy doesn't care what the category is.
        class_id = np.argmax(y)
        class_predict = np.argmax(np.bincount(r[ct], minlength=10))
        if(class_predict == class_id):
            if(correct_H_ct == 0):
                correct_H=[H]
            else:
                correct_H = np.append(correct_H, [H], axis=0)
            correct_H_ct = correct_H_ct + 1
        else:
            if(incorrect_H_ct == 0):
                incorrect_H = [H]
            else:
                incorrect_H = np.append(incorrect_H, [H], axis=0)
            incorrect_H_ct = incorrect_H_ct + 1

    print(np.histogram(correct_H))
    print(np.histogram(incorrect_H))




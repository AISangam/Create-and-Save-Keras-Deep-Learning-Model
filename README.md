# Create-and-Save-Keras-Deep-Learning-Model   

![keras_summary](https://user-images.githubusercontent.com/35392729/57950412-c7c64280-7904-11e9-939b-a56efd4d9412.png)

## Why to use Keras to create deep learning models??  

Keras uses either tensorflow or theano at the backend. If one have to change the backend, just go to the home directory in case of ubuntu (cd ~). Please locate the file .keras there. If you cannot find it please type in the terminal ls -a. Please go the directory cd .keras. You will see file keras.json. Please open the file with the favourite editor such as nano or vim. You will see the output similar to what is shown below

![keras_json](https://user-images.githubusercontent.com/35392729/57950836-014b7d80-7906-11e9-9de8-340236cca8e6.png)

<b>Let us see some of the advantage of using keras</b>
<ol>
    <li> Keras is very easy to use as compared to tensorflow or other deep learning librarary.</li>
    <li> It gives very friendly look and hence is easy to start with.</li>
    <li> It is easy to define activation function or layers in keras. </li>
</ol>

## Keras model save    
Given that deep learning models can take hours, days and even weeks to train, it is important to know how to save and load them from disk. Keras separates the concerns of saving your model architecture and saving your model weights. Model weights are saved to HDF5 format. Please see the below code to know it better

```
# creating the checkpoint directory. This is same where we need to save the model
checkpoint = dirName
# join the filename to the checkpoint directory
file_path = os.path.join(checkpoint,
                    "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")

# model saving provided only best model in the desired epochs are saved.
checkpoint = ModelCheckpoint(file_path, monitor='val_acc',
                             verbose=1, save_best_only=True, mode='max')
# this will be passed during training
callbacks_list = [checkpoint]
```
## This code is used to save the checkpoints but there are other ways also

When training deep learning models, the checkpoint is the weights of the model. These weights can be used to make predictions as is, or used as the basis for ongoing training. In other words one can save the model in json or yaml and weights in hdf5 as below
```
# serialize model to JSON
#  the keras model which is trained is defined as 'model' in this example
model_json = model.to_json()

# write the model in json format
# seralize modelin json
with open("model_in.json", "w") as json_file:
    json_file.write(model_json)

OR

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model_in.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)


# serialize weights to HDF5
model.save_weights("model_num.h5")
```  

## How to run this file  

<ul>
<li> First of all one need to install requirements_apt.txt. This is an important step because this is used to install dependencies to plot the model in form of image. To do this step, please follow the below step.</li></ul>  

```
cat requirements_apt.txt | xargs sudo apt-get install -y
```
<ul>
  <li> Now run the pip requirement. I hope you should work on virtualenv by using below command virtualenv --python python3 **name_of_env**. If you face any problem in creating the virtualenv, please do follow https://bit.ly/2Q1UAo0. Please do follow below step for this operation.</li></ul>  
  
  ```
  pip3 install -r requirements_pip.txt
  ```
 <ul>
  <li> Please note that once you have installed and created virtualenv, it is important to activate it before installing pip requirement. Please take care of this point. Now please run the python file.</li></ul>  
  
  ```
  python Keras_create_save.py
  ```
  
  ### References for help  
  <ol>
  <li> https://www.aisangam.com/ </li>
  <li> https://www.aisangam.com/blog/ </li> </ol>
  
  Thanks for reading this.
  
  
  

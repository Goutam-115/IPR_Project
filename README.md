Instructions to run the model

• To download the dependencies , execute the below command. require-ment.txt is uploaded in github.

pip install -r requirements.txt


• The github repository have 2 .py files, for two implemented model,
– Refiner.py
– Generator.py

• To run the model, specify the proper train data and test data directories in the dataloading section in the code. (specified in the code)

• Internet access must be on to get the preloaded vgg-16 weights for the CSRnet model.
– If internet access is not available, you can download the VGG-16 pre-trained weights manually from the following link: 'https://download.pytorch.org/models/vgg16-397923af.pth'. After downloaded, there is a separate commented piece of code that takes the local path of the VGG-16 weights. You can un-comment that part of the code and update the file path to point to the downloaded .pth file. Also, comment out the line that loads the weights online.(CSRNet() declaration without vgg-16 path)

• Run the below command to run, execute and evaluate the model.

 python3 Refiner.py

 python3 Generator.py

- You can increase number of epochs for training in the model implementation.

-Drive link for dataset : 'https://drive.google.com/drive/folders/1sb6srQlqVsEkB8Xt28BHN0muu6Bo4tqq?usp=sharing'

reddit
classification and generation experiments on the reddit dataset
 
 
Steps for training:

1: clean up the data with make_data.py which creates a new csv.

2: if there are pretrained embeddings, run the new csv through 
    filter embeddings to create embedding matrix. make sure to have the same 
    parameters on this file, and the model.
    
3: now, it is ready to train. you can train the auto encoder, or
    the classifier. it saves the model which is the same for both, 
    so it is possible to train the auto encoder, and then train 
    that model for classification. Of course the models can be trained multiple 
    times for the same thing. Be weary, because the 
    hyperparameters must be the same in order for you to do this.
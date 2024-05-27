import argparse

import numpy as np
from torchinfo import summary
from scipy import stats
from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data)
    print(xtrain.shape, xtest.shape, ytrain.shape)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    print(xtrain.shape, xtest.shape, ytrain.shape)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # normalize data
    mu = np.mean(xtrain, 0, keepdims=True)
    std = np.std(xtrain, 0, keepdims=True)
    xtrain = normalize_fn(xtrain, mu, std)
    xtest = normalize_fn(xtest, mu, std)
    z_scores = stats.zscore(xtrain, axis=0)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    xtrain = xtrain[filtered_entries]
    ytrain = ytrain[filtered_entries]

    #xtrain = append_bias_term(xtrain)
    #xtest = append_bias_term(xtest)
    
    """
    xtrain = xtrain[:6000]
    ytrain = ytrain[:6000]
    xtest = xtest[:1000]
    """
    
    # Make a validation set
    if not args.test:
        ### WRITE YOUR CODE HERE
        validation_ratio = 0.2  # for example, 20% of the data will be used as the validation set
        num_validation_samples = int(validation_ratio * xtrain.shape[0])
        
         # Shuffle the data before splitting (optional but recommended)
        indices_x = np.arange(xtrain.shape[0])
        np.random.shuffle(indices_x)
        xtrain = xtrain[indices_x] 
        ytrain = ytrain[indices_x] 
        
        
        # Split the data
        xtrain = xtrain[num_validation_samples:]
        ytrain = ytrain[num_validation_samples:]
        xtest = xtrain[:num_validation_samples]
        ytest = ytrain[:num_validation_samples]  

        
    ### WRITE YOUR CODE HERE to do any other data processing
    

    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data
        pca_obj.find_principal_components(xtrain)
        xtrain = pca_obj.reduce_dimension(xtrain)

    ## 3. Initialize the method you want to use.
    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        ### WRITE YOUR CODE HERE
        model = MLP(xtest.shape[1], n_classes)
    elif args.nn_type == "cnn":
        ### WRITE YOUR CODE HERE
        print(xtest.shape)
        model = CNN(1, n_classes)
    elif args.nn_type == "transformer":
        ### WRITE YOUR CODE HERE
        model = MyViT((1, 28, 28), 7, 4, 16, 4, n_classes)   #CHANGE VALUES CAN MAKE IT BETTER


    summary(model)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    if not args.test:
        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
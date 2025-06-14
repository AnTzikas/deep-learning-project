'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
MODIFIED FOR TENSORFLOW 2.x and MODERN KERAS
'''
# import numpy as np
# import keras
# from keras.models import Model, load_model
# from keras.layers import Embedding, Input, Dense, Flatten, multiply, concatenate
# from keras.optimizers import Adagrad, Adam, SGD, RMSprop
# from keras.regularizers import l2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, multiply, concatenate
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import argparse
import GMF
import MLP

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--topK', type=int, default=10,
                        help='Top K.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]//2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()


def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # --- Embedding layers ---
    # MF embeddings
    mf_embedding_user = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  embeddings_initializer = 'glorot_uniform', embeddings_regularizer = l2(reg_mf))
    mf_embedding_item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  embeddings_initializer = 'glorot_uniform', embeddings_regularizer = l2(reg_mf))   
    # MLP embeddings
    # NOTE: The original code used integer division. In Python 3, use // for clarity.
    mlp_embedding_user = Embedding(input_dim = num_users, output_dim = layers[0]//2, name = "mlp_embedding_user",
                                  embeddings_initializer = 'glorot_uniform', embeddings_regularizer = l2(reg_layers[0]))
    mlp_embedding_item = Embedding(input_dim = num_items, output_dim = layers[0]//2, name = 'mlp_embedding_item',
                                  embeddings_initializer = 'glorot_uniform', embeddings_regularizer = l2(reg_layers[0]))   
    
    # --- MF (GMF) part ---
    mf_user_latent = Flatten()(mf_embedding_user(user_input))
    mf_item_latent = Flatten()(mf_embedding_item(item_input))
    # CHANGE: 'merge' is deprecated. Use 'multiply'.
    mf_vector = multiply([mf_user_latent, mf_item_latent])

    # --- MLP part ---
    mlp_user_latent = Flatten()(mlp_embedding_user(user_input))
    mlp_item_latent = Flatten()(mlp_embedding_item(item_input))
    # CHANGE: 'merge' is deprecated. Use 'concatenate'.
    mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])
    # for idx in range(1, num_layer):
    #     # CHANGE: 'W_regularizer' is deprecated. Use 'kernel_regularizer'.
    #     layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
    #     mlp_vector = layer(mlp_vector)
    for idx in range(1, num_layer):
        layer_name = "layer%d" %idx
        # --- THIS IS THE NEW LOGIC ---
        # If it's the last MLP layer, give it a specific name for distillation
        if idx == num_layer - 1:
            layer_name = "final_mlp_layer"
        # --- END NEW LOGIC ---
            
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name=layer_name)
        mlp_vector = layer(mlp_vector)    
    # --- Concatenate MF and MLP parts ---
    # CHANGE: 'merge' is deprecated. Use 'concatenate'.
    predict_vector = concatenate([mf_vector, mlp_vector], name='neumf_concat')
    
    # --- Final prediction layer ---
    # CHANGE: 'init' is deprecated. Use 'kernel_initializer'.
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = "prediction")(predict_vector)
    
    # CHANGE: 'input' is deprecated. Use 'inputs'. 'output' -> 'outputs'.
    model = Model(inputs=[user_input, item_input], outputs=prediction)
    
    return model

# def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # This function is designed to work with models loaded by keras.models.load_model()
    
    # Load MF embeddings
    # NOTE: The layer names in GMF_model are 'user_embedding' and 'item_embedding'.
    # We map them to 'mf_embedding_user' and 'mf_embedding_item' in our NeuMF model.
    model.get_layer('mf_embedding_user').set_weights(gmf_model.get_layer('user_embedding').get_weights())
    model.get_layer('mf_embedding_item').set_weights(gmf_model.get_layer('item_embedding').get_weights())
    
    # Load MLP embeddings
    # NOTE: The layer names in MLP_model are also 'user_embedding' and 'item_embedding'.
    # We map them to 'mlp_embedding_user' and 'mlp_embedding_item' in our NeuMF model.
    model.get_layer('mlp_embedding_user').set_weights(mlp_model.get_layer('user_embedding').get_weights())
    model.get_layer('mlp_embedding_item').set_weights(mlp_model.get_layer('item_embedding').get_weights())
    
    # Load MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
        
    # Load prediction weights
    # The GMF and MLP models both have a 'prediction' layer.
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    
    # The new prediction layer's weights are a concatenation of the GMF and MLP prediction weights.
    # The NeuMF prediction layer input is [mf_vector, mlp_vector], so the weights must match.
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    # The new bias is an average of the GMF and MLP biases.
    new_b = (gmf_prediction[1] + mlp_prediction[1]) / 2.0
    
    model.get_layer('prediction').set_weights([new_weights, new_b])    
    return model
# NEW, CORRECTED load_pretrain_model function
def load_pretrain_model(model, gmf_weights, mlp_weights, num_layers):
    # MF embeddings
    # The GMF model has 2 layers of weights (user, item) and a prediction layer
    gmf_user_embeddings = [gmf_weights[0]]  # Keras expects a list
    gmf_item_embeddings = [gmf_weights[1]]
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)
    
    # MLP embeddings
    # The MLP model has user/item embeddings, N dense layers, and a prediction layer
    mlp_user_embeddings = [mlp_weights[0]]
    mlp_item_embeddings = [mlp_weights[1]]
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)
    
    # MLP layers
    # Each dense layer has a weight matrix and a bias vector
    for i in range(1, num_layers):
        # Calculate the correct index for the mlp_weights list
        mlp_layer_weights = [mlp_weights[2*i], mlp_weights[2*i+1]]
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
        
    # Prediction weights
    # The GMF prediction layer weights are the last two elements of the gmf_weights list
    gmf_prediction = [gmf_weights[-2], gmf_weights[-1]]
    # The MLP prediction layer weights are the last two elements of the mlp_weights list
    mlp_prediction = [mlp_weights[-2], mlp_weights[-1]]
    
    # Concatenate the weights for the final NeuMF prediction layer
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = (gmf_prediction[1] + mlp_prediction[1]) / 2.0
    
    model.get_layer('prediction').set_weights([new_weights, new_b])    
    return model

def get_train_instances(train, num_negatives, num_items):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            # CHANGE: 'train.has_key' is Python 2. Use 'in' for Python 3.
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

# In NeuMF.py

if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain
    topK = args.topK
    # topK = 10
    evaluation_threads = 1
    print("NeuMF arguments: %s " %(args))
    
    # --- FILENAME CHANGE: Save as .npy ---
    model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.npy' %(args.dataset, mf_dim, args.layers, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
    #model.summary()
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    
    # Load pretrain model (This part is for GMF/MLP pretraining, not what we're focused on now)
    if mf_pretrain != '' and mlp_pretrain != '':
        # ... (your pre-training logic for NeuMF can stay here) ...
        print("Pre-training logic would go here if specified.")
        
    # Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1

    # --- NO SAVE COMMAND HERE ---
    # We must NOT save the untrained model.
        
    # Training model
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives, num_items)
        
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    # --- THE ONLY SAVE COMMAND ---
                    # This saves the weights of the BEST performing model as a .npy file
                    np.save(model_out_file, model.get_weights())

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best NeuMF model weights are saved to %s" %(model_out_file))

# if __name__ == '__main__':
#     args = parse_args()
#     epochs = args.epochs
#     batch_size = args.batch_size
#     mf_dim = args.num_factors
#     layers = eval(args.layers)
#     reg_mf = args.reg_mf
#     reg_layers = eval(args.reg_layers)
#     num_negatives = args.num_neg
#     learning_rate = args.lr
#     learner = args.learner
#     verbose = args.verbose
#     mf_pretrain = args.mf_pretrain
#     mlp_pretrain = args.mlp_pretrain
#     topK = args.topK
#     # topK = 10
#     evaluation_threads = 1
#     print("NeuMF arguments: %s " %(args))
#     model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.npy' %(args.dataset, mf_dim, args.layers, time())

#     # Loading data
#     t1 = time()
#     dataset = Dataset(args.path + args.dataset)
#     train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
#     num_users, num_items = train.shape
#     print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
#           %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
#     # Build model
#     model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
#     #model.summary()
#     if learner.lower() == "adagrad": 
#         model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
#     elif learner.lower() == "rmsprop":
#         model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
#     elif learner.lower() == "adam":
#         model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
#     else:
#         model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    
#     # Load pretrain model
#     # if mf_pretrain != '' and mlp_pretrain != '':
#     #     # CHANGE: Load the entire saved model instead of just weights. This is more robust.
#     #     gmf_model = load_model(mf_pretrain)
#     #     mlp_model = load_model(mlp_pretrain)
#     #     model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
#     #     print("Load pretrained GMF (%s) and MLP (%s) models done." %(mf_pretrain, mlp_pretrain))
#     # NEW (CORRECTED) CODE
# # FINAL CORRECTED CODE for NeuMF.py pre-training block

#     # FINAL, CORRECTED PRE-TRAINING BLOCK FOR NeuMF.py

#     # FINAL, CORRECTED PRE-TRAINING BLOCK FOR NeuMF.py

# # Load pretrain model
#     if mf_pretrain != '' and mlp_pretrain != '':
#         # Load the weights from the .npy files
#         gmf_weights = np.load(mf_pretrain, allow_pickle=True)
#         mlp_weights = np.load(mlp_pretrain, allow_pickle=True)
    
#         # Pass the NumPy weight arrays directly to the loading function
#         model = load_pretrain_model(model, gmf_weights, mlp_weights, len(layers))
#         print("Load pretrained GMF (%s) and MLP (%s) models done." %(mf_pretrain, mlp_pretrain))
        
#     # Init performance
#     (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
#     hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
#     print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
#     best_hr, best_ndcg, best_iter = hr, ndcg, -1
#     if args.out > 0:
#         # CHANGE: save the whole model instead of just weights
#         model.save(model_out_file)
        
#     # Training model
#     for epoch in range(epochs):
#         t1 = time()
#         # Generate training instances
#         user_input, item_input, labels = get_train_instances(train, num_negatives, num_items)
        
#         # Training
#         # CHANGE: 'nb_epoch' is deprecated. Use 'epochs'.
#         hist = model.fit([np.array(user_input), np.array(item_input)], #input
#                          np.array(labels), # labels 
#                          batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
#         t2 = time()
        
#         # Evaluation
#         if epoch % verbose == 0:
#             (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
#             hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
#             print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
#                   % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
#             if hr > best_hr:
#                 best_hr, best_ndcg, best_iter = hr, ndcg, epoch
#                 if args.out > 0:
#                     # Corrected line
#                     np.save(model_out_file, model.get_weights())
#                     #model.save_weights(model_out_file, overwrite=True) 
#                     #model.save(model_out_file, overwrite=True)

#     print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
#     if args.out > 0:
#         print("The best NeuMF model is saved to %s" %(model_out_file))

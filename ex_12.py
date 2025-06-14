# ex_12.py (Final Version with all 3 Distillation Techniques)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Forces CPU execution

import numpy as np
import pandas as pd
import argparse
from time import time

import tensorflow as tf
tf.enable_eager_execution()

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.train import AdamOptimizer

from Dataset import Dataset
from evaluate import evaluate_model
from NeuMF import get_model as get_neumf_model
from MLP import get_model as get_mlp_model
import ast

def parse_args():
    parser = argparse.ArgumentParser(description="Run Knowledge Distillation for NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-100k', help='Choose a dataset.')
    parser.add_argument('--technique', nargs='?', default='response', 
                        help='Distillation technique: response, feature, mlp_student, or relation.') # <-- Added relation
    parser.add_argument('--teacher_model', type=str, required=True, 
                        help='Path to the pre-trained teacher NeuMF weights file (.npy format).')
    parser.add_argument('--epochs', type=int, default=11, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--num_neg', type=int, default=4, help='Number of negatives.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='Weight for the standard student loss term.')
    parser.add_argument('--out', type=int, default=0, help='Whether to save the best student model.')
    parser.add_argument('--layers', nargs='?', default='[32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]//2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    return parser.parse_args()

def get_train_instances(train, num_negatives, num_items):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    for (u, i) in train.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(1.0)
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0.0)
    return np.array(user_input, dtype=np.int32), \
           np.array(item_input, dtype=np.int32), \
           np.array(labels, dtype=np.float32)

def main():
    args = parse_args()
    
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    
    print(f"Building and loading teacher model from {args.teacher_model}")
    teacher_mf_dim = 32
    teacher_layers = [64, 32, 16, 8]
    teacher_model = get_neumf_model(num_users, num_items, mf_dim=teacher_mf_dim, layers=teacher_layers, reg_layers=[0,0,0,0], reg_mf=0)
    teacher_weights = np.load(args.teacher_model, allow_pickle=True)
    teacher_model.set_weights(teacher_weights)
    teacher_model.trainable = False 
    
    print(f"Using distillation technique: {args.technique}")
    student_model = None
    teacher_feature_extractor = None

    student_layers = ast.literal_eval(args.layers)
    student_reg_layers = ast.literal_eval(args.reg_layers)

    if args.technique in ['response', 'feature', 'relation']: # <-- Added relation
        student_model = get_neumf_model(num_users, num_items, mf_dim=args.num_factors, layers=student_layers, reg_layers=student_reg_layers, reg_mf=0)
        if args.technique == 'feature':
            teacher_feature_extractor = Model(inputs=teacher_model.input, outputs=teacher_model.get_layer('final_mlp_layer').output)
            student_feature_extractor = Model(inputs=student_model.input, outputs=student_model.get_layer('final_mlp_layer').output)
        elif args.technique == 'relation':
            # For relation, we need the MLP item embeddings
            teacher_feature_extractor = Model(inputs=teacher_model.input[1], outputs=teacher_model.get_layer('mlp_embedding_item').output)
            student_feature_extractor = Model(inputs=student_model.input[1], outputs=student_model.get_layer('mlp_embedding_item').output)
    elif args.technique == 'mlp_student':
        student_model = get_mlp_model(num_users, num_items, layers=student_layers, reg_layers=student_reg_layers)
    else:
        raise ValueError("Invalid technique specified.")
        
    print("\n--- Model Complexity ---")
    print(f"Teacher Model Parameters: {teacher_model.count_params():,}")
    print(f"Student Model Parameters: {student_model.count_params():,}")

    optimizer = AdamOptimizer(learning_rate=args.lr)
    bce = tf.keras.losses.binary_crossentropy
    mse = tf.keras.losses.mean_squared_error

    best_hr, best_ndcg, best_iter = 0, 0, -1
    student_model_out_file = f'Pretrain/{args.dataset}_student_{args.technique}.npy'
    topK = 10

    for epoch in range(args.epochs):
        t1 = time()
        user_input, item_input, labels = get_train_instances(train, args.num_neg, num_items)
        train_dataset = tf.data.Dataset.from_tensor_slices(((user_input, item_input), labels)).shuffle(len(user_input)).batch(args.batch_size)

        total_loss = 0
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            user_batch, item_batch = x_batch
            with tf.GradientTape() as tape:
                y_pred_student_raw = student_model(x_batch, training=True)
                y_pred_teacher_raw = teacher_model(x_batch, training=False)
                y_pred_student = tf.squeeze(y_pred_student_raw)
                y_pred_teacher = tf.squeeze(y_pred_teacher_raw)
                
                student_loss = bce(y_batch, y_pred_student)
                distill_loss = 0
                
                if args.technique == 'response' or args.technique == 'mlp_student':
                    distill_loss = bce(y_pred_teacher, y_pred_student)
                elif args.technique == 'feature':
                    teacher_features = teacher_feature_extractor(x_batch)
                    student_features = student_feature_extractor(x_batch)
                    distill_loss = mse(teacher_features, student_features)
                elif args.technique == 'relation':
                    # Get item embeddings for the current batch
                    H_t_batch = tf.squeeze(teacher_feature_extractor(item_batch))
                    H_s_batch = tf.squeeze(student_feature_extractor(item_batch))
                    # Compute similarity matrices
                    S_t = tf.matmul(H_t_batch, H_t_batch, transpose_b=True)
                    S_s = tf.matmul(H_s_batch, H_s_batch, transpose_b=True)
                    distill_loss = mse(S_t, S_s)

                combined_loss_per_sample = args.alpha * student_loss + (1 - args.alpha) * distill_loss
                batch_loss = tf.reduce_mean(combined_loss_per_sample)

            grads = tape.gradient(batch_loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
            total_loss += batch_loss
        
        avg_loss = total_loss / (step + 1)
        t2 = time()

        t_eval_start = time()
        (hits, ndcgs) = evaluate_model(student_model, testRatings, testNegatives, topK, num_thread=1)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        t_eval_end = time()
        
        print(f'Epoch {epoch+1} [{t2-t1:.1f}s]: loss = {avg_loss:.4f}, HR = {hr:.4f}, NDCG = {ndcg:.4f} [eval {t_eval_end-t_eval_start:.1f}s]')
        
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            if args.out > 0:
                np.save(student_model_out_file, student_model.get_weights())
    
    print(f"\nEnd. Best Epoch {best_iter+1}: HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}.")
    if args.out > 0:
        print(f"The best student model weights are saved to {student_model_out_file}")

if __name__ == '__main__':
    main()
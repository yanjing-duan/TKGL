import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np

from dataset import Graph_Tune_Parameter_Dataset_Fusion_DA
from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score, f1_score

import os
from model import BertModel, PredictModelFusionDA

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
keras.backend.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

small = {'name': 'Small', 'num_layers': 3, 'num_heads': 2, 'd_model': 128, 'path': 'small_weights', 'addH': True}
medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights', 'addH': True}
medium3 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 4, 'd_model': 256, 'path': 'medium_weights3',
           'addH': True}
large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 512, 'path': 'large_weights', 'addH': True}



def cl_loss_cal(x1, x2, T=0.1):
    x1 = tf.math.l2_normalize(x1, axis=1)
    x2 = tf.math.l2_normalize(x2, axis=1)
    sim_matrix = tf.matmul(x1, x2, transpose_b=True)
    sim_matrix = tf.clip_by_value(sim_matrix, -1.0 + 1e-8, 1.0 - 1e-8)
    scaled_sim = sim_matrix / T
    scaled_sim = tf.clip_by_value(scaled_sim, -50.0, 50.0)
    sim_matrix_exp = tf.exp(scaled_sim)
    pos_sim = tf.linalg.diag_part(sim_matrix_exp)
    denom = tf.reduce_sum(sim_matrix_exp, axis=1) - pos_sim
    denom = tf.maximum(denom, 1e-8)
    loss1 = pos_sim / denom
    loss1 = tf.clip_by_value(loss1, 1e-8, 1.0)
    loss = -tf.reduce_mean(tf.math.log(loss1))
    return loss

def dare_gram_loss(H1, H2):
    dare_threshold = args["dare_threshold"]
    dare_tradeoff_angle = args["dare_tradeoff_angle"]
    dare_tradeoff_scale = args["dare_tradeoff_scale"]

    b, p = H1.shape

    ones = tf.ones([b, 1], dtype=H1.dtype)
    A = tf.concat([ones, H1], axis=1)
    B = tf.concat([ones, H2], axis=1)

    cov_A = tf.matmul(A, A, transpose_a=True)
    cov_B = tf.matmul(B, B, transpose_a=True)

    s_A = tf.linalg.svd(cov_A, compute_uv=False)
    s_B = tf.linalg.svd(cov_B, compute_uv=False)

    s_A_sg = tf.stop_gradient(s_A)
    s_B_sg = tf.stop_gradient(s_B)
    
    eigen_A = tf.cumsum(s_A_sg) / tf.reduce_sum(s_A_sg)
    eigen_B = tf.cumsum(s_B_sg) / tf.reduce_sum(s_B_sg)

    T_A = tf.cond(eigen_A[1] > dare_threshold,
                 lambda: tf.stop_gradient(eigen_A[1]),
                 lambda: tf.constant(dare_threshold, dtype=eigen_A.dtype))
    
    T_B = tf.cond(eigen_B[1] > dare_threshold,
                 lambda: tf.stop_gradient(eigen_B[1]),
                 lambda: tf.constant(dare_threshold, dtype=eigen_B.dtype))

    indices_A = tf.where(eigen_A <= T_A)
    index_A = tf.cond(tf.size(indices_A) > 0,
                     lambda: indices_A[-1, 0],
                     lambda: tf.constant(0, dtype=tf.int64))
    
    indices_B = tf.where(eigen_B <= T_B)
    index_B = tf.cond(tf.size(indices_B) > 0,
                     lambda: indices_B[-1, 0],
                     lambda: tf.constant(0, dtype=tf.int64))
    
    k = tf.maximum(index_A, index_B)

    rcond_A = tf.stop_gradient(s_A_sg[k] / s_A_sg[0])
    rcond_B = tf.stop_gradient(s_B_sg[k] / s_B_sg[0])
    
    A_pinv = tf.linalg.pinv(cov_A, rcond=rcond_A)
    B_pinv = tf.linalg.pinv(cov_B, rcond=rcond_B)

    cos_sim = tf.reduce_sum(A_pinv * B_pinv, axis=0) / (
        tf.norm(A_pinv, axis=0) * tf.norm(B_pinv, axis=0) + 1e-6
    )
    cos_dist = tf.reduce_sum(tf.abs(cos_sim - 1.0)) / tf.cast(p + 1, cos_sim.dtype)
    svd_dist = tf.reduce_sum(tf.abs(s_A_sg[:k] - s_B_sg[:k])) / tf.cast(k, s_A_sg.dtype)
    loss = dare_tradeoff_angle * cos_dist + dare_tradeoff_scale * svd_dist

    return loss

def main(seed=7, arch = medium3, pretraining = 'NO', trained_epoch = 8, task = 'BBBP', max_epoch = 100, batch_size = 64, learning_rate=10e-5):

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    addH = arch['addH']

    dff = d_model * 2
    vocab_size = 17
    da_init_epoch = args["init_epochs"]

    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)
    dataset_class = Graph_Tune_Parameter_Dataset_Fusion_DA(train_path="./split_data/{}/{}/train_BioassaySA.csv".format(task, seed), val_path="./split_data/{}/{}/val_BioassaySA.csv".format(task, seed), test_path="./split_data/{}/{}/test_BioassaySA.csv".format(task, seed), smiles_field='smiles', label_field='Label',addH=addH)
    source_dataset, _, target_dataset = dataset_class.get_data(batch_size=batch_size, shuffle_val=True, drop_remainder_val=True)
    _, test_dataset, val_dataset = dataset_class.get_data(batch_size=512, shuffle_val=False, drop_remainder_val=False)

    x, adjoin_matrix, y, x_bioassay, x_sa = next(iter(test_dataset.take(1)))
    print("x_bioassay.shape: ", x_bioassay.shape, "x_sa.shape: ", x_sa.shape)
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = PredictModelFusionDA(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                        dropout_rate = args['dropout_rate'], dense_dropout=args['dense_dropout'], KE_dim=x_bioassay.shape[1], SA_dim=x_sa.shape[1], hidden_dim=args['fnn_hidden_dim'],
                        num_hidden_layers=args['num_fnn_hidden_layers'], feat_views=3)

    if pretraining == "self_supervised_pretraining":
        print("task: {}, self-supervise_pre-trained".format(task))
        if not os.path.exists('./'+arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch)):
            temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
            pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
            temp.load_weights('./'+arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
            temp.encoder.save_weights('./'+arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
            del temp

        pred, _, _, _, _ = model(x=x, adjoin_matrix=adjoin_matrix, mask=mask, x_KE=x_bioassay, x_SA=x_sa, training=True)
        model.encoder.load_weights('./'+arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        print('load_self_supervised_pretraining_weights')

    elif pretraining == "NO":
        print("No pre-training")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    auc= -10
    stopping_monitor = 0

    len_source = len(source_dataset)
    len_target = len(target_dataset)
    print('len_source:', len_source, 'len_target:', len_target)
    steps_per_epoch = max(len_source, len_target)

    for epoch in range(max_epoch):
        accuracy_object = tf.keras.metrics.BinaryAccuracy()
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        total_loss = 0.0
        label_loss_accum = 0.0
        cl_loss_accum = 0.0
        da_loss_accum = 0.0

        source_iter = iter(source_dataset)
        target_iter = iter(target_dataset)

        for step in range(steps_per_epoch):
            with tf.GradientTape() as tape:
                try:
                    source_x, source_adjoin_matrix, source_y, source_x_bioassay, source_x_sa = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_dataset)
                    source_x, source_adjoin_matrix, source_y, source_x_bioassay, source_x_sa = next(source_iter)
                source_seq = tf.cast(tf.math.equal(source_x, 0), tf.float32)
                source_mask = source_seq[:, tf.newaxis, tf.newaxis, :]
                source_preds, _, _, source_x_list, source_logits = model(x=source_x, adjoin_matrix=source_adjoin_matrix, mask=source_mask, x_KE=source_x_bioassay, x_SA=source_x_sa, training=True)
                label_loss = loss_object(source_y,source_preds)

                cl_loss = 0.0
                num_pairs = 0
                for i in range(len(source_x_list)):
                    for j in range(i + 1, len(source_x_list)):
                        cl_loss_ = cl_loss_cal(source_x_list[i], source_x_list[j])
                        cl_loss += cl_loss_
                        num_pairs += 1
                cl_loss = cl_loss / num_pairs
                
                if epoch >= da_init_epoch:
                    try:
                        target_x, target_adjoin_matrix, _, target_x_bioassay, target_x_sa = next(target_iter)
                    except StopIteration:
                        target_iter = iter(target_dataset)
                        target_x, target_adjoin_matrix, _, target_x_bioassay, target_x_sa = next(target_iter)

                    target_seq = tf.cast(tf.math.equal(target_x, 0), tf.float32)
                    target_mask = target_seq[:, tf.newaxis, tf.newaxis, :]
                    _, _, _, _, target_logits = model(x=target_x, adjoin_matrix=target_adjoin_matrix, mask=target_mask, x_KE=target_x_bioassay, x_SA=target_x_sa, training=True)
                    da_loss = dare_gram_loss(source_logits, target_logits)
                    loss = label_loss + args['alpha'] * cl_loss + da_loss * args['beta']
                else:
                    da_loss = 0.0
                    loss = label_loss + args['alpha'] * cl_loss

                total_loss += loss
                label_loss_accum += label_loss
                cl_loss_accum += cl_loss
                if epoch >= da_init_epoch:
                    da_loss_accum += da_loss

                accuracy_object.update_state(source_y,source_preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
                
        avg_total_loss = total_loss / steps_per_epoch
        avg_label_loss = label_loss_accum / steps_per_epoch
        avg_cl_loss = cl_loss_accum / steps_per_epoch
        avg_da_loss = da_loss_accum / steps_per_epoch if epoch >= da_init_epoch else 0.0
        print('epoch: {}, total loss: {:.4f}, label_loss: {:.4f}, cl_loss: {:.4f}, da_loss: {:.4f}, accuracy: {:.4f}'.format(epoch, avg_total_loss.numpy().item(), avg_label_loss.numpy().item(), avg_cl_loss.numpy().item(), avg_da_loss.numpy().item(), accuracy_object.result().numpy().item()))

        y_true = []
        y_preds = []

        for x, adjoin_matrix, y, x_bioassay, x_sa in val_dataset:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds, _, _, _, _ = model(x=x, adjoin_matrix=adjoin_matrix, mask=mask, x_KE=x_bioassay, x_SA=x_sa, training=False)
            y_true.append(y.numpy())
            y_preds.append(preds.numpy())
        y_true = np.concatenate(y_true,axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds,axis=0).reshape(-1)
        y_preds = tf.sigmoid(y_preds).numpy()
        auc_new = roc_auc_score(y_true,y_preds)

        val_accuracy = keras.metrics.binary_accuracy(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
        print('val auc:{:.4f}'.format(auc_new), 'val accuracy:{:.4f}'.format(val_accuracy))

        if auc_new > auc:
            auc = auc_new
            stopping_monitor = 0

            if pretraining == "self_supervised_pretraining":
                model.save_weights('model_weights/self_{}_fusion_da_alpha{}_beta{}_dropoutrate{}_densedropout{}_learningrate{}_batchsize{}_fnnhiddendim{}_numfnnhiddenlayers{}.h5'.format(task, args['alpha'], args['beta'], args['dropout_rate'], args['dense_dropout'], args['learning_rate'], args['batch_size'], args['fnn_hidden_dim'], args['num_fnn_hidden_layers']))
            elif pretraining == "NO":
                model.save_weights('model_weights/{}_fusion_da_alpha{}_beta{}_dropoutrate{}_densedropout{}_learningrate{}_batchsize{}_fnnhiddendim{}_numfnnhiddenlayers{}.h5'.format(task, args['alpha'], args['beta'], args['dropout_rate'], args['dense_dropout'], args['learning_rate'], args['batch_size'], args['fnn_hidden_dim'], args['num_fnn_hidden_layers']))
        else:
            stopping_monitor += 1
        print('best val auc: {:.4f}'.format(auc))
        if stopping_monitor>0:
            print('stopping_monitor:',stopping_monitor)
        if stopping_monitor==20:
            break

    y_true = []
    y_preds = []
    all_KE_attention_weights = []
    all_SA_attention_weights = []

    if pretraining == "self_supervised_pretraining":
        model.load_weights('model_weights/self_{}_fusion_da_alpha{}_beta{}_dropoutrate{}_densedropout{}_learningrate{}_batchsize{}_fnnhiddendim{}_numfnnhiddenlayers{}.h5'.format(task, args['alpha'], args['beta'], args['dropout_rate'], args['dense_dropout'], args['learning_rate'], args['batch_size'], args['fnn_hidden_dim'], args['num_fnn_hidden_layers']))
    elif pretraining == "NO":
        model.load_weights('model_weights/{}_fusion_da_alpha{}_beta{}_dropoutrate{}_densedropout{}_learningrate{}_batchsize{}_fnnhiddendim{}_numfnnhiddenlayers{}.h5'.format(task, args['alpha'], args['beta'], args['dropout_rate'], args['dense_dropout'], args['learning_rate'], args['batch_size'], args['fnn_hidden_dim'], args['num_fnn_hidden_layers']))

    for x, adjoin_matrix, y, x_bioassay, x_sa in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds, KE_attention_weights, SA_attention_weights, _, _ = model(x=x, adjoin_matrix=adjoin_matrix, mask=mask, x_KE=x_bioassay, x_SA=x_sa, training=False)
        y_true.append(y.numpy())
        y_preds.append(preds.numpy())
        all_KE_attention_weights.append(KE_attention_weights.numpy())
        all_SA_attention_weights.append(SA_attention_weights.numpy())
    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
    all_KE_attention_weights = np.concatenate(all_KE_attention_weights, axis=0)
    all_SA_attention_weights = np.concatenate(all_SA_attention_weights, axis=0)

    y_preds = tf.sigmoid(y_preds).numpy()
    y_preds_label = y_preds.reshape(-1) > 0.5
    y_preds_label = y_preds_label.astype(int)

    test_auc = roc_auc_score(y_true, y_preds)
    test_accuracy = keras.metrics.binary_accuracy(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
    test_mcc = matthews_corrcoef(y_true.reshape(-1), y_preds_label)
    test_ba = balanced_accuracy_score(y_true.reshape(-1), y_preds_label)
    test_f1 = f1_score(y_true.reshape(-1), y_preds_label)

    print('test auc:{:.4f}'.format(test_auc), 'test accuracy:{:.4f}'.format(test_accuracy), 'test mcc:{:.4f}'.format(test_mcc))

    return test_auc, test_accuracy, test_mcc, test_ba, test_f1

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Two-Step-Pretraining')

    parser.add_argument('--pretraining', type= str, default='NO', help='Whether to load the weights of a trained model and which model to load. Choose one from ["self_supervised_pretraining", "NO"], corresponding to the self-supervised and no pretraining.')
    parser.add_argument('--trained_epoch', type=int, default=100,
                        help='which epoch of the pretrained model to use (default: 100)')
    parser.add_argument('--task', type=str, default='BBBP',
                        help='the fine-tuning task (default: BBBP)')
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='the maximum training epoch of the fine-tuning (default: 100)')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout rate (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size of the training set (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=10e-5,
                        help='learning rate (default: 10e-5)')
    parser.add_argument('--dense_dropout', type=float, default=0.15,
                        help='dense dropout (default: 0.15)')
    parser.add_argument('--fnn_hidden_dim', type=int, default=300,
                        help='hidden size of FFN (default: 300)')
    parser.add_argument('--num_fnn_hidden_layers', type=int, default=1,
                        help='the number of layers of the feed forward neural network (default: 1)')
    parser.add_argument('--alpha', type=float, default=0.08,
                        help='alpha (default: 0.08) the balance ratio of label_loss, cl_loss and da_loss, this is before cl_loss')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta (default: 1) the balance ratio of label_loss, cl_loss and da_loss, this is before da_loss, this is loss_trade_off')
    parser.add_argument('--init_epochs', type=int, default=0,
                        help='init_epochs (default: 0) the epoch begin to add da_loss')
    parser.add_argument('--dare_threshold', type=float, default=0.96,
                        help='dare_threshold (default: 0.96)')
    parser.add_argument('--dare_tradeoff_angle', type=float, default=0.05,
                        help='dare_tradeoff_angle (default: 0.05)')
    parser.add_argument('--dare_tradeoff_scale', type=float, default=0.001,
                        help='dare_tradeoff_scale (default: 0.001)')




    args = parser.parse_args().__dict__
    print(args)

    results = pd.DataFrame()
    test_auc_list = []
    test_acc_list = []
    test_mcc_list = []
    test_ba_list = []
    test_f1_list = []

    pretraining = args['pretraining'] # pretraining：["self_supervised_pretraining", 'NO']
    trained_epoch = args['trained_epoch']
    task = args['task']
    max_epoch = args['max_epoch']
    dropout_rate = args['dropout_rate']
    batch_size = args['batch_size']
    learning_rate = args['learning_rate']
    dense_dropout = args['dense_dropout']

    arch = medium3
    for seed in [7,17,27,37,47,57,67,77,87,97]:
        print("seed:",seed)
        test_auc, test_accuracy, test_mcc, test_ba, test_f1 = main(seed=seed, arch=arch, pretraining=pretraining, trained_epoch=trained_epoch, task=task, max_epoch=max_epoch, batch_size=batch_size, learning_rate=learning_rate)

        test_auc_list.append(test_auc)
        test_acc_list.append(test_accuracy)
        test_mcc_list.append(test_mcc)
        test_ba_list.append(test_ba)
        test_f1_list.append(test_f1)

    results["test_auc"] = test_auc_list
    results["test_acc"] = test_acc_list
    results["test_mcc"] = test_mcc_list
    results["test_ba"] = test_ba_list
    results["test_f1"] = test_f1_list

    print("task:", task)
    print("dropout_rate:", dropout_rate)
    print("batch_size:", batch_size)
    print("learning_rate:", learning_rate)
    print("dense_dropout:", dense_dropout)
    print(results)

    if pretraining == "self_supervised_pretraining":
        results.to_csv('model_results/self_{}_fusion_da_alpha{}_beta{}_dropoutrate{}_densedropout{}_learningrate{}_batchsize{}_fnnhiddendim{}_numfnnhiddenlayers{}.csv'.format(task, args['alpha'], args['beta'], args['dropout_rate'], args['dense_dropout'], args['learning_rate'], args['batch_size'], args['fnn_hidden_dim'], args['num_fnn_hidden_layers']), index=False)
    elif pretraining == "NO":
        results.to_csv('model_results/{}_fusion_da_alpha{}_beta{}_dropoutrate{}_densedropout{}_learningrate{}_batchsize{}_fnnhiddendim{}_numfnnhiddenlayers{}.csv'.format(task, args['alpha'], args['beta'], args['dropout_rate'], args['dense_dropout'], args['learning_rate'], args['batch_size'], args['fnn_hidden_dim'], args['num_fnn_hidden_layers']), index=False)
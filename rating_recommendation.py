import pandas as pd
import tensorflow as tf
import time
import pickle
from libreco.algorithms import ALS, NCF, SVD, ItemCF, RNN4Rec, SVDpp, UserCF
from libreco.data import DatasetPure, split_by_ratio_chrono



# Preprocessing 

movie_df = pd.read_csv('ml-25m/movies.csv')
movie_df = movie_df.iloc[:10000,:]  #Taking Small Subset of whole data
movie_df['genres'] = movie_df['genres'].str.replace('|', ' ')
movie_df['genres'] = movie_df['genres'].str.lower()
movie_df['title'] = movie_df['title'].str.extract(r'(^.+)\s\(\d{4}\)')  # To remove the year from movie name
movie_df['title'] = movie_df['title'].str.split(',').str[0]  # To correct the movie name as some movie name given is wrong


data = pd.read_csv('ml-25m/ratings.csv')
data = data.iloc[:100000, :]  #Taking Small subset of whole data
data.columns = ["user", "item", "label", "time"]



# This is used to reset after each algorithm is run
def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)



#Funtion to get movie name through it movie ID
def get_movie_name(movie_id_list):
    movie_name_list = []
    for movie_id in movie_id_list:
        movie_name = movie_df[movie_df["movieId"] == movie_id]['title'].tolist()[0]
        movie_name_list.append(movie_name)
    return movie_name_list



# Taking User input
user = int(input("Enter the user id: "))
item = int(input("Enter the item id: "))
n_rec = int(input("Enter the number of recommendations: "))



data = data  
train_data, eval_data = split_by_ratio_chrono(data, test_size=0.2)
train_data, data_info = DatasetPure.build_trainset(train_data)
eval_data = DatasetPure.build_evalset(eval_data)
metrics = ["rmse", "mae", "r2"]




#Defining Algorithms 

reset_state("SVD")
svd = SVD(
    "rating",
    data_info,
    embed_size=16,
    n_epochs=3,
    lr=0.001,
    reg=None,
    batch_size=256,
    num_neg=1,
)
svd.fit(
    train_data,
    neg_sampling=False,
    verbose=2,
    shuffle=True,
    eval_data=eval_data,
    metrics=metrics,
)


reset_state("SVD++")
svdpp = SVDpp(
    task="rating",
    data_info=data_info,
    embed_size=16,
    n_epochs=3,
    lr=0.001,
    reg=None,
    batch_size=256,
)
svdpp.fit(
    train_data,
    neg_sampling=False,
    verbose=2,
    shuffle=True,
    eval_data=eval_data,
    metrics=metrics,
)


reset_state("NCF")
ncf = NCF(
    "rating",
    data_info,
    embed_size=16,
    n_epochs=3,
    lr=0.001,
    lr_decay=False,
    reg=None,
    batch_size=256,
    num_neg=1,
    use_bn=True,
    dropout_rate=None,
    hidden_units=(128, 64, 32),
    tf_sess_config=None,
)
ncf.fit(
    train_data,
    neg_sampling=False,
    verbose=2,
    shuffle=True,
    eval_data=eval_data,
    metrics=metrics,
)


reset_state("RNN4Rec")
rnn = RNN4Rec(
    "rating",
    data_info,
    rnn_type="lstm",
    embed_size=16,
    n_epochs=2,
    lr=0.001,
    lr_decay=False,
    hidden_units=(16, 16),
    reg=None,
    batch_size=256,
    num_neg=1,
    dropout_rate=None,
    recent_num=10,
    tf_sess_config=None,
)
rnn.fit(
    train_data,
    neg_sampling=False,
    verbose=2,
    shuffle=True,
    eval_data=eval_data,
    metrics=metrics,
)



reset_state("ALS")
als = ALS(
    task="rating",
    data_info=data_info,
    embed_size=16,
    n_epochs=2,
    reg=5.0,
    alpha=10,
    use_cg=False,
    n_threads=1,
    seed=42,
)
als.fit(
    train_data,
    neg_sampling=False,
    verbose=2,
    shuffle=True,
    eval_data=eval_data,
    metrics=metrics,
)


reset_state("user_cf")
user_cf = UserCF(
    task="rating",
    data_info=data_info,
    k_sim=20,
    sim_type="cosine",
    mode="invert",
    num_threads=4,
    min_common=1,
)
user_cf.fit(
    train_data,
    neg_sampling=False,
    verbose=2,
    eval_data=eval_data,
    metrics=metrics,
)


reset_state("item_cf")
item_cf = ItemCF(
    task="rating",
    data_info=data_info,
    k_sim=20,
    sim_type="pearson",
    mode="invert",
    num_threads=1,
    min_common=1,
)
item_cf.fit(
    train_data,
    neg_sampling=False,
    verbose=2,
    eval_data=eval_data,
    metrics=metrics,
)

movie_name = movie_df[movie_df["movieId"] == item]['title'].tolist()[0]

print("SVD:")
print(f"Prediction for user {user} for the movie '{movie_name}' is: {svd.predict(user=user, item=item)}")
print(f"Top {n_rec} recommendations for user {user}: {get_movie_name(svd.recommend_user(user=user, n_rec=n_rec)[user])}")

print("SVD++:")
print(f"Prediction for user {user} for the movie '{movie_name}' is: {svdpp.predict(user=user, item=item)}")
print(f"Top {n_rec} recommendations for user {user}: {get_movie_name(svdpp.recommend_user(user=user, n_rec=n_rec)[user])}")

print("NCF:")
print(f"Prediction for user {user} for the movie '{movie_name}' is: {ncf.predict(user=user, item=item)}")
print(f"Top {n_rec} recommendations for user {user}: {get_movie_name(ncf.recommend_user(user=user, n_rec=n_rec)[user])}")

print("RNN4Rec:")
print(f"Prediction for user {user} for the movie '{movie_name}' is: {rnn.predict(user=user, item=item)}")
print(f"Top {n_rec} recommendations for user {user}: {get_movie_name(rnn.recommend_user(user=user, n_rec=n_rec)[user])}")

print("ALS:")
print(f"Prediction for user {user} for the movie '{movie_name}' is: {als.predict(user=user, item=item)}")
print(f"Top {n_rec} recommendations for user {user}: {get_movie_name(als.recommend_user(user=user, n_rec=n_rec)[user])}")

print("UserCF:")
print(f"Prediction for user {user} for the movie '{movie_name}' is: {user_cf.predict(user=user, item=item)}")
print(f"Top {n_rec} recommendations for user {user}: {get_movie_name(user_cf.recommend_user(user=user, n_rec=n_rec)[user])}")

print("ItemCF:")
print(f"Prediction for user {user} for the movie '{movie_name}' is: {item_cf.predict(user=user, item=item)}")
print(f"Top {n_rec} recommendations for user {user}: {get_movie_name(item_cf.recommend_user(user=user, n_rec=n_rec)[user])}")

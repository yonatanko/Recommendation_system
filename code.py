import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import lsqr
from scipy import sparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from math import log10 as log


def load():
    test_Df = pd.read_csv('test.csv')
    user_artist_Df = pd.read_csv('user_artist.csv')
    return test_Df, user_artist_Df


def build_ranked_dicts(user_artists_Df, type_of_data):
    users_dict = {}
    artists_dict = {}
    ranks = {}
    ranks_values = []
    for i, row in user_artists_Df.iterrows():
        if row["userID"] not in users_dict.keys():
            users_dict[row["userID"]] = [row["artistID"]]
        else:
            users_dict[row["userID"]].append(row["artistID"])

        if row["artistID"] not in artists_dict.keys():
            artists_dict[row["artistID"]] = [row["userID"]]
        else:
            artists_dict[row["artistID"]].append(row["userID"])

        if type_of_data != "final test":
            if row["artistID"] not in ranks.keys():
                ranks[row["artistID"]] = {row["userID"]: row["weight"]}
                ranks_values.append(row["weight"])
            else:
                ranks[row["artistID"]][row["userID"]] = row["weight"]
                ranks_values.append(row["weight"])

    return users_dict, artists_dict, ranks, list(users_dict.keys()), list(artists_dict.keys())


def Build_A(rows, ranks_dict, users, artists):
    num_users = len(users)
    num_artists = len(artists)
    users_index = dict(zip(users, list(range(num_users))))
    j_lst = list(range(num_users, num_users + num_artists))
    artists_index = dict(zip(artists, j_lst))

    num_cols = num_users + num_artists
    A = dok_matrix((rows, num_cols))
    index = 0
    for artist in ranks_dict.keys():
        for user in ranks_dict[artist].keys():
            A[index, users_index[user]] = 1
            A[index, artists_index[artist]] = 1
            index += 1

    return A, users_index, artists_index


def Build_b(ranks_dict):
    inner_dicts = [inner_dict for inner_dict in ranks_dict.values()]
    ranks = [item for sublist in inner_dicts for item in sublist.values()]
    ranks = np.array(ranks)
    r_avg = ranks.mean()
    b = np.array([x - r_avg for x in ranks])
    return b, r_avg, ranks


def extract_biases(x, users, artists, user_indexes, artists_indexes):
    users_biases = {}
    artists_biases = {}
    for user in users:
        users_biases[user] = x[user_indexes[user]]
    for artist in artists:
        artists_biases[artist] = x[artists_indexes[artist]]

    return users_biases, artists_biases


def build_regularized_mat(A, b, lambda_reg):
    y = np.array([lambda_reg ** (1 / 2)] * A.shape[1])
    regularized_part = sparse.spdiags(y, 0, A.shape[1], A.shape[1])

    zeros = np.zeros(A.shape[1])
    new_mat = sparse.vstack([A, regularized_part])
    new_vec = np.concatenate((b, zeros))

    return new_mat, new_vec


def train_test_split_manual(user_artists_df):
    user_count = user_artists_df.value_counts("userID").rename_axis('userID').reset_index(name='counts')
    artist_count = user_artists_df.value_counts("artistID").rename_axis('artistID').reset_index(name='counts')
    users_to_train = user_count[user_count["counts"] <= 10]
    artists_to_train = artist_count[artist_count["counts"] <= 10]
    users_to_train_list = users_to_train["userID"].tolist()
    artist_to_train_list = artists_to_train["artistID"].tolist()
    train_df = user_artists_df.loc[(user_artists_df["userID"].isin(users_to_train_list)) | (
        user_artists_df["artistID"].isin(artist_to_train_list))]
    train_df = train_df.reset_index()[["userID", "artistID", "weight"]]

    others_df = user_artists_df.merge(train_df.drop_duplicates(), on=['userID', 'artistID', 'weight'], how='left',
                                      indicator=True)
    others_df = others_df[others_df['_merge'] == "left_only"].reset_index()[["userID", "artistID", "weight"]]
    train_2, test_df = train_test_split(others_df, test_size=0.4)
    train_df = pd.concat([train_df, train_2]).reset_index()[["userID", "artistID", "weight"]]
    print(f"train size: {train_df.shape[0] / user_artists_df.shape[0] * 100}%")
    print(f"test size: {test_df.shape[0] / user_artists_df.shape[0] * 100}%")

    return train_df, test_df


def regularized_process(train_df, optimal_lambda):
    users_dict, artists_dict, ranks, users, artists = build_ranked_dicts(train_df, "train final")
    A, users_indexes, artists_indexes = Build_A(train_df.count()["userID"], ranks, users, artists)
    b, r_avg, ranks_list = Build_b(ranks)

    # train
    if optimal_lambda != 0:
        A,b = build_regularized_mat(A, b, optimal_lambda)
    x = lsqr(A,b)[0]
    users_biases, artists_biases = extract_biases(x, users, artists, users_indexes, artists_indexes)

    return users_biases, artists_biases, r_avg, ranks, min(ranks_list), max(
        ranks_list), artists, users, users_dict, artists_dict


def build_R_tilda(users_biases, artists_biases, r_avg, R_ranks, min_rank, max_rank):
    R_tilda_ranks = R_ranks.copy()
    for artist in R_ranks.keys():
        for user in R_ranks[artist].keys():
            hat_rank = r_avg + users_biases[user] + artists_biases[artist]
            if hat_rank > max_rank:  # clipping to original range
                R_tilda_ranks[artist][user] = R_ranks[artist][user] - max_rank
            else:
                if hat_rank < min_rank:
                    R_tilda_ranks[artist][user] = R_ranks[artist][user] - min_rank
                else:  # in original range of ranks
                    R_tilda_ranks[artist][user] = R_ranks[artist][user] - hat_rank

    return R_tilda_ranks


def R_tilda_based_mat(R_tilda_ranks):
    new_df = pd.DataFrame(R_tilda_ranks)
    new_df = new_df.fillna(0)
    mat = new_df.to_numpy()
    return mat


def build_similarity_mat(R_tilda_matrix):
    dot_products = np.matmul(R_tilda_matrix, R_tilda_matrix.T)
    indices_list = np.argwhere(dot_products != 0)
    similarity_mat = np.zeros((R_tilda_matrix.shape[0], R_tilda_matrix.shape[0]))
    for indices in indices_list:
        i = indices[0]
        j = indices[1]
        if i == j or similarity_mat[i][j] != 0:
            continue
        artists = np.array([R_tilda_matrix[i, :], R_tilda_matrix[j, :]]).T
        ranked_both = artists[np.all(artists != 0, axis=1)]
        if ranked_both.shape[0] == 1:
            continue
        else:
            row_i = ranked_both[:, 0]
            row_j = ranked_both[:, 1]
            similarity_mat[i][j] = dot_products[i][j] / (np.linalg.norm(row_i) * np.linalg.norm(row_j))
            similarity_mat[j][i] = similarity_mat[i][j]
    return similarity_mat


def find_k_closest_neighbours(threshold, similarity_matrix, artists):
    k_nearest_neighbours = {}
    for i in range(similarity_matrix.shape[0]):
        artist_row = np.array(similarity_matrix[i, :])
        closest_indices = np.argwhere(np.abs(artist_row) >= threshold).flatten().tolist()
        if i in closest_indices:
            closest_indices.remove(i)
        k_nearest_neighbours[artists[i]] = [(artists[index], similarity_matrix[i][index]) for index in closest_indices]

    return k_nearest_neighbours


def neighbourhood_part_prediction(k_nearest_dict, R_tilda_matrix, users_dict, user, artist, train_users, train_artists):
    sum_up = 0
    sum_low = 0
    ranked_by_user = []

    for tuple_artist_distance in k_nearest_dict[artist]:
        if tuple_artist_distance[0] in users_dict[user]:
            ranked_by_user.append(tuple_artist_distance)

    if len(ranked_by_user) == 0:  # no common user
        return 0
    else:
        for curr_artist_distance in ranked_by_user:
            curr_artist_index = train_artists.index(curr_artist_distance[0])
            distance = curr_artist_distance[1]
            user_index = train_users.index(user)
            tilda_rank = R_tilda_matrix[curr_artist_index][user_index]
            sum_up += distance * tilda_rank
            sum_low += abs(distance)

    if sum_low == 0:
        return 0
    else:
        return sum_up / sum_low


def calculate_prediction(R_tilda_matrix, artists, artists_dict, users_dict, k_nearest_dict, r_avg, users_biases,
                         artists_biases, min_rank, train_users, train_artists):
    preds = {}
    ranks = []
    for artist in artists:
        for user in artists_dict[artist]:
            if user not in train_users and artist not in train_artists:
                # both user and artist do not appear in train set
                rank = r_avg
            else:  # if only one of them appear
                if user not in train_users:
                    rank = r_avg + artists_biases[artist]
                else:
                    if artist not in train_artists:
                        rank = r_avg + users_biases[user]
                    else:  # both appear
                        first_part = r_avg + users_biases[user] + artists_biases[artist]
                        neighbourhood_part = neighbourhood_part_prediction(k_nearest_dict, R_tilda_matrix, users_dict,
                                                                           user, artist, train_users, train_artists)
                        rank = first_part + neighbourhood_part
                        if rank < min_rank:
                            rank = min_rank

            if artist not in preds.keys():
                preds[artist] = {user: 10 ** rank}
                ranks.append(10 ** rank)
            else:
                preds[artist][user] = 10 ** rank
                ranks.append(10 ** rank)

    return preds, ranks


def build_log_train(user_artists_df):
    logged_train = user_artists_df.copy()
    logged_train["weight"] = np.log10(logged_train["weight"])
    return logged_train


def calculate_predictions_for_test(train_df, test_df, threshold, lambda_reg):
    users_biases, artists_biases, r_avg, R_ranks, min_rank, max_rank, artists, users, users_dict, artists_dict = \
        regularized_process(train_df, lambda_reg)
    R_tilda_ranks = build_R_tilda(users_biases, artists_biases, r_avg, R_ranks, min_rank, max_rank)
    R_tilda_matrix = R_tilda_based_mat(R_tilda_ranks)
    R_tilda_matrix = R_tilda_matrix.T
    similarity_mat = build_similarity_mat(R_tilda_matrix)
    users_dict_test, artists_dict_test, ranks_test, users_test, artists_test = build_ranked_dicts(test_df, "final test")

    # train
    k_nearest_neighbours_dict = find_k_closest_neighbours(threshold, similarity_mat, artists)

    # test
    predictions, ranks = calculate_prediction(R_tilda_matrix, artists_test, artists_dict_test, users_dict_test,
                                              k_nearest_neighbours_dict, r_avg, users_biases,
                                              artists_biases, min_rank, users,artists)

    return predictions, ranks


def main():

    test_Df, user_artist_Df = load()
    train_df = build_log_train(user_artist_Df)
    preds, ranks = calculate_predictions_for_test(train_df, test_Df, 0.7, 10)
    test_Df["weight"] = ranks
    test_Df.to_csv("test_output.csv", index = False)


if __name__ == "__main__":
    main()

#***********************************************************************************************************************

# not in use functions

def calculate_RMSE(ranks, artists, artists_dict, preds):
    count = 0
    sum = 0
    for artist in artists:
        for user in artists_dict[artist]:
            sum += (preds[artist][user] - ranks[artist][user]) ** 2
            count += 1

    return (sum / count) ** (1 / 2)


def lambda_cross_validation(num_iter, lambdas, title):  # cross validation to find optimal lambda
    avg_train_reg = []
    avg_test_reg = []
    first = True
    test_Df, user_artist_Df = load()
    for i in range(num_iter):
        start = time.time()
        train_df, test_df = train_test_split_manual(user_artist_Df)
        train_rmse_list_reg, test_rmse_list_reg = regularized_process_for_cross_validation(train_df, test_df, lambdas)

        if first:
            avg_train_reg = train_rmse_list_reg
            avg_test_reg = test_rmse_list_reg
            first = False
        else:
            avg_train_reg = [sum(x) for x in zip(avg_train_reg, train_rmse_list_reg)]
            avg_test_reg = [sum(x) for x in zip(avg_test_reg, test_rmse_list_reg)]
        print(f"time of {i + 1} iteration: {round(time.time() - start, 3)} seconds")

    avg_train_reg = [x / num_iter for x in avg_train_reg]
    avg_test_reg = [x / num_iter for x in avg_test_reg]

    figure, (plotter1, plotter2) = plt.subplots(2)
    figure.tight_layout(pad=4.0)
    plot_func(avg_train_reg, plotter1, "train", lambdas, "lambda")
    plot_func(avg_test_reg, plotter2, "test", lambdas, "lambda")
    plt.savefig(title)


def plot_func(regularized_rmse_list, plotter, group_type, hyper_parameter_list, hyper_parameter):
    plotter.scatter(hyper_parameter_list, regularized_rmse_list, color="orange")
    plotter.plot(hyper_parameter_list, regularized_rmse_list, label="regularized " + group_type + " RMSE")

    plotter.set_title(group_type)
    plotter.set_xlabel(hyper_parameter)
    plotter.set_ylabel("RMSE")


def calculate_RMSE_for_baseline(user_biases, artists_biases, ranks, artists, artists_dict, r_avg, min_rank, max_rank):
    count = 0
    sum = 0
    for artist in artists:
        for user in artists_dict[artist]:
            real_rank = ranks[artist][user]
            count += 1
            # if clipping is needed
            if r_avg + user_biases[user] + artists_biases[artist] > max_rank:
                sum += (max_rank - real_rank) ** 2
            else:
                if r_avg + user_biases[user] + artists_biases[artist] < min_rank:
                    sum += (min_rank - real_rank) ** 2
                else:  # in range of max and min
                    sum += (r_avg + user_biases[user] + artists_biases[artist] - real_rank) ** 2

    return (sum / count) ** (1 / 2)


def regularized_process_for_cross_validation(train_df, test_df, lambdas):
    train_rmse_list = []
    test_rmse_list = []

    users_dict, artists_dict, ranks, users, artists = build_ranked_dicts(train_df, "train final")
    A, users_index, artists_index = Build_A(train_df.count()["userID"], ranks, users, artists)
    b, r_avg, ranks_list = Build_b(ranks)

    users_dict_test, artists_dict_test, ranks_test, users_test, artists_test = build_ranked_dicts(test_df,
                                                                                                  "not final test")
    for lambda_reg in lambdas:
        # train
        new_mat, new_vec = build_regularized_mat(A, b, lambda_reg)
        x = lsqr(new_mat, new_vec)[0]
        users_biases, artists_biases = extract_biases(x, users, artists, users_index,artists_index)
        train_rmse_list.append(
            calculate_RMSE_for_baseline(users_biases, artists_biases, ranks, artists, artists_dict, r_avg,
                                        min(ranks_list), max(ranks_list)))

        # test
        test_rmse_list.append(
            calculate_RMSE_for_baseline(users_biases, artists_biases, ranks_test, artists_test, artists_dict_test,
                                        r_avg, min(ranks_list), max(ranks_list)))

    return train_rmse_list, test_rmse_list



def threshold_cross_validaion(train_df, test_df, threshold_list, lambda_reg):
    rmse_train = []
    rmse_test = []

    users_biases, artists_biases, r_avg, R_ranks, min_rank, max_rank, artists, users, users_dict, artists_dict = regularized_process(
        train_df, lambda_reg)
    R_tilda_ranks = build_R_tilda(users_biases, artists_biases, r_avg, R_ranks, min_rank, max_rank)
    R_tilda_matrix = R_tilda_based_mat(R_tilda_ranks)
    R_tilda_matrix = R_tilda_matrix.T
    similarity_mat = build_similarity_mat(R_tilda_matrix)
    users_dict_test, artists_dict_test, ranks_test, users_test, artists_test = build_ranked_dicts(test_df,
                                                                                                  "not final test")
    for threshold in threshold_list:
        print(f"threshold: {threshold}")
        k_nearest_neighbours_dict = find_k_closest_neighbours(threshold, similarity_mat, artists)
        preds, ranks = calculate_prediction(R_tilda_matrix, artists, artists_dict, users_dict,
                                            k_nearest_neighbours_dict, users, r_avg, users_biases, artists_biases,
                                            min_rank, artists)

        rmse_train.append(calculate_RMSE(R_ranks, artists, artists_dict, preds))

        # test
        preds, ranks_test_returned = calculate_prediction(R_tilda_matrix, artists_test, artists_dict_test,
                                                          users_dict_test, k_nearest_neighbours_dict, users_test, r_avg,
                                                          users_biases, artists_biases, min_rank, artists)
        rmse_test.append(calculate_RMSE(ranks_test, artists_test, artists_dict_test, preds))

    return rmse_train, rmse_test


def simulation_process_for_neighbourhood(num_iter, threshold_list, lambda_reg, title):
    avg_train = []
    avg_test = []
    first = True
    test_Df, user_artist_Df = load()
    for i in range(num_iter):
        start = time.time()
        train_df, test_df = train_test_split_manual(user_artist_Df)
        train_rmse_list, test_rmse_list = threshold_cross_validaion(train_df, test_df, threshold_list, lambda_reg)

        if first:
            avg_train = train_rmse_list
            avg_test = test_rmse_list
            first = False
        else:
            avg_train = [sum(x) for x in zip(avg_train, train_rmse_list)]
            avg_test = [sum(x) for x in zip(avg_test, test_rmse_list)]
        print(f"time of {i + 1} iteration: {round(time.time() - start, 3)} seconds")

    avg_train = [x / num_iter for x in avg_train]
    avg_test = [x / num_iter for x in avg_test]

    figure, (plotter1, plotter2) = plt.subplots(2)
    figure.tight_layout(pad=4.0)
    plot_func(avg_train, plotter1, "train", threshold_list, "threshold")
    plot_func(avg_test, plotter2, "test", threshold_list, "threshold")
    plt.savefig(title)



def calculate_final_loss(ranks, artists, artists_dict, preds):
    sum = 0
    for artist in artists:
        for user in artists_dict[artist]:
            sum += (log(preds[artist][user]) - ranks[artist][user]) ** 2

    return sum


def predict_and_calculate_losses(train_df, test_df, threshold, lambda_reg):
    # train
    users_biases, artists_biases, r_avg, R_ranks, min_rank, max_rank, artists, users, users_dict, artists_dict = regularized_process(train_df, lambda_reg)
    R_tilda_ranks = build_R_tilda(users_biases, artists_biases, r_avg, R_ranks, min_rank, max_rank)
    R_tilda_matrix = R_tilda_based_mat(R_tilda_ranks)
    R_tilda_matrix = R_tilda_matrix.T
    similarity_mat = build_similarity_mat(R_tilda_matrix)
    users_dict_test, artists_dict_test, ranks_test, users_test, artists_test = build_ranked_dicts(test_df,
                                                                                                  "not final test")
    k_nearest_neighbours_dict = find_k_closest_neighbours(threshold, similarity_mat, artists)
    predictions, ranks = calculate_prediction(R_tilda_matrix, artists, artists_dict, users_dict,
                                              k_nearest_neighbours_dict, r_avg, users_biases, artists_biases,
                                              min_rank, users,artists)
    loss_train = calculate_final_loss(R_ranks, artists, artists_dict, predictions)

    # test
    predictions, ranks = calculate_prediction(R_tilda_matrix, artists_test, artists_dict_test, users_dict_test,
                                              k_nearest_neighbours_dict, r_avg, users_biases,
                                              artists_biases, min_rank, users, artists)
    loss_test = calculate_final_loss(ranks_test, artists_test, artists_dict_test, predictions)

    return loss_train, loss_test


def simulation_process_final_check(num_iter, threshold, lambda_reg):
    sum_train = 0
    sum_test = 0
    test_Df, user_artist_Df = load()
    for i in range(num_iter):
        train_df, test_df = train_test_split_manual(user_artist_Df)
        train_df = build_log_train(train_df)
        test_df = build_log_train(test_df)
        train_loss, test_loss = predict_and_calculate_losses(train_df, test_df, threshold, lambda_reg)
        sum_train += train_loss
        sum_test += test_loss

    print(f"avg of {num_iter} runs train loss: {sum_train / num_iter}")
    print(f"avg of {num_iter} tuns test loss: {sum_test / num_iter}")

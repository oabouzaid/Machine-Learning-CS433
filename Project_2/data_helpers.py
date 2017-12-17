from helpers import *


def create_csv_submission(test_data_path, output_path, predictions):
    """create csv submission for the test data using the predictions."""

    def deal_line(line):
        row_col_id, _ = line.split(',')
        row, col = row_col_id.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row)-1, int(col)-1, row_col_id

    with open(test_data_path, "r") as f_in:
        test_data = f_in.read().splitlines()
        fieldnames = test_data[0].split(",")
        test_data = test_data[1:]
    
    with open(output_path, 'w') as f_out:
        writer = csv.DictWriter(f_out, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for line in test_data:
            item, user, user_item_id = deal_line(line)
            prediction = predictions[item, user]
            writer.writerow({
                fieldnames[0]: user_item_id,
                fieldnames[1]: prediction
            })


def split_data(ratings, split=0.1, seed=998):
    """
    Source: Lab 10 Solutions
    split the ratings to training data and test data.
    """
    
    # set seed
    np.random.seed(988)

    nz_items, nz_users = ratings.nonzero()

    # create sparse matrices to store the data
    num_rows, num_cols = ratings.shape
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))

    for user in set(nz_users):
        row, col = ratings[:, user].nonzero()
        selects = np.random.choice(row, size=int(len(row) * split))
        non_selects = list(set(row) - set(selects))

        train[non_selects, user] = ratings[non_selects, user]
        test[selects, user] = ratings[selects, user]
        
    return train, test
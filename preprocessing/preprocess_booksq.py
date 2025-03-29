import pandas as pd

# Parameters
## Filter
min_i = 10
min_u = 10
## Split
training_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

#################
# Load the data #
#################

print("Loading data...", flush=True)

dir_path = "../data/books_q/"
original_path = dir_path + "original_data"
final_path = dir_path
df= pd.read_csv(original_path + "/data.csv", sep=",")
df = df[['rating', 'user_id', 'isbn', 'Category']]

df = df[(df["rating"] >= 4) | (df["rating"] == -1)] # Remove interactions corresponding to low (< 4) ratings

print(sorted(df))

####################
# K-core filtering #
####################

category_user_ids = set(df[df['Category'] != -1].user_id)
df = df[df['user_id'].isin(category_user_ids)]

print("Head\n", df.head())

# Filter out items with less than min_i interactions
print("Filtering out items...", flush=True)
dist_df = df['isbn'].value_counts()

dist_df = dist_df[dist_df >= min_i]
filtered_item_ids = dist_df.keys()
df = df[df['isbn'].isin(filtered_item_ids)]

# Filter out users with less than min_u interactions
print("Filtering out users...", flush=True)
dist_df = df['user_id'].value_counts()
dist_df = dist_df[dist_df >= min_u]
filtered_user_ids = dist_df.keys()
df = df[df['user_id'].isin(filtered_user_ids)]


filtered_user_ids = set(df['user_id']) # Update list of users filtered to remove those with no interaction
filtered_item_ids = set(df['isbn']) # Update list of items filtered to remove those with no interaction
print("Number of users:", len(filtered_user_ids))
print("Number of items:", len(filtered_item_ids))


training_tuples = []
validation_tuples = []
test_tuples = []
user_list = set(df['user_id'])
item_list = set(df['isbn'])
user_dict = {u: user_id for (user_id, u) in enumerate(user_list)}
item_dict = {p: item_id for (item_id, p) in enumerate(item_list)}

for (u, user_id) in user_dict.items():
    user_query_df = df[df['user_id'] == u].sort_values(by=['rating'])
    if user_id % 1000 == 0:
        print("Number of users processed: " + str(user_id), flush=True)
    n_interaction = user_query_df.shape[0]
    n_test = int(test_ratio * n_interaction)
    n_validation = int(validation_ratio * n_interaction)
    n_training = n_interaction - n_validation - n_test

    for interaction_count, (row, interaction) in enumerate(user_query_df.iterrows()):
        item_id = item_dict[interaction['isbn']] # Item interacted
        rating = interaction['rating'] # Rating of the interaction or -1 if no rating
        rating = "-" if rating == -1 else str(rating)

        #added tag interaction by 
        tag = interaction['Category'] # ID of the tag or -1 if no tag
        tag_text = "-" if tag == -1 else str(tag)
        tag_text = tag_text.replace('[', '').replace(']', '')
        tag_text = '' + tag_text.replace('\t', ' ') + ''

        # Process the query and add it to training, validation or test set
        if interaction_count < n_training: # Training set
            training_tuples.append((user_id, item_id, rating, tag_text))
        elif interaction_count >= n_training and interaction_count < n_training + n_validation: # Validation set
            validation_tuples.append((user_id, item_id, rating,tag_text))
        else: # Test set
            test_tuples.append((user_id, item_id, rating,tag_text))

n_user = len(user_list)
n_item = len(item_dict)

print("Training", len(training_tuples), flush=True)
print("Validation", len(validation_tuples), flush=True)
print("Test", len(test_tuples), flush=True)


##########################
# Save preprocessed data #
##########################

print("Saving preprocessed data...", flush=True)

# Save data size file
data_size_path = final_path + "/data_size.txt"
with open(data_size_path, "w+", encoding='utf-8') as f:
    f.write(str(n_user) + "\t" + str(n_item) + "\n")

# Save training file
training_path = final_path + "/train.txt"
with open(training_path, "w+", encoding='utf-8') as f:
    f.write("user_id\titem_id\trating\tquery\n")
    for (user_id, item_id, rating, tag_text) in training_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + rating +"\t"+str(tag_text)+"\n")


# Save validation file
training_path = final_path + "/valid.txt"
with open(training_path, "w+", encoding='utf-8') as f:
    f.write("user_id\titem_id\trating\tquery\n")
    for (user_id, item_id, rating, tag_text) in validation_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + rating +"\t"+str(tag_text)+"\n")


# Save test file
training_path = final_path + "/test.txt"
with open(training_path, "w+", encoding='utf-8') as f:
    f.write("user_id\titem_id\trating\tquery\n")
    for (user_id, item_id, rating, tag_text) in test_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + rating +"\t"+str(tag_text)+"\n")

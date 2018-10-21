##################################

# EVALUATE ON TEST DATA

##################################



# We pick the last sequence for each id in the test data
seq_array_test_last = [
        test_df[test_df['id']==id][sequence_cols].values[-sequence_length:]  for id in test_df['id'].unique() 
        if len(test_df[test_df['id']==id]) >= sequence_length
    ]


seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
print("seq_array_test_last")
print(seq_array_test_last)
print(seq_array_test_last.shape)

# Similarly, we pick the labels


y_mask = [len(test_df[test_df['id']==id]) >= sequence_length for id in test_df['id'].unique()]
label_array_test_last = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)
model = load_model(model_path)

# test metrics
scores_test = model.predict(seq_array_test_last)

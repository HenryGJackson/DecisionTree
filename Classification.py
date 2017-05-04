import pandas as pd
from network import network

def normalise_data(df):
    for i, j in enumerate(df):
        if j == "Species":
            # print ("String")
            continue
        # print (df[j])
        maxi = max(df[j])
        df[j] = df[j] / maxi
        # print (df[j])
        # break
    return df

def array_length_check(pass_array, fail_array):
    if len(pass_array) < 1 | len(fail_array) < 1:
        print("Error: neuron->array_length_check: pass and fail training sets must have length > 0")
        if len(pass_array) < 1:
            print("Error source --> pass set")
        if len(fail_array) < 1:
            print("Error source --> fail set")
        return False
    elif len(pass_array) != len(fail_array):
        print("Error: neuron->array_length_check: Both the training arrays must have equal length")
        return False
    else:
        return True


df = pd.read_csv("Iris.csv")
df = normalise_data(df)
net = network(5)
nwin_df = df[df["Species"] == "Iris-setosa"].reset_index()
nlose_df = df[df["Species"] != "Iris-setosa"].reset_index()
nwin_df = nwin_df.drop("Species", axis=1)
nlose_df = nlose_df.drop("Species", axis=1)
print(nwin_df)
# exit()
if len(nwin_df) != len(nlose_df):
    # print("Cutting Dataframe",len(win_df), len(lose_df))
    if len(nwin_df) > len(nlose_df):
        nwin_df = nwin_df[:len(nlose_df)]
    else:
        nlose_df = nlose_df[:len(nwin_df)]
# print(win_df)
# print(lose_df)
print("Cut Dataframe", len(nwin_df), len(nlose_df))

net.create_network(nwin_df, nlose_df)
print("network created")
net.visualise_tree()
# test_df =
# event = df.iloc[130].drop("Species")
# print(event)
# answer = net.seed.pass_on(df.iloc[130].drop("Species"))
exit()
# print(answer)
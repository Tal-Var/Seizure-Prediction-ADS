def toBinary(x):
    if x != 1: return 0;
    else: return 1;
y = y['y'].apply(toBinary)
y = pd.DataFrame(data=y)
y

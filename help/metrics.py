def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)

    n = y.size
    n_correct = 0
    for i in range(n):
        if y[i] == y_hat[i]:
            n_correct+=1

    return round(n_correct*100/n, 2)

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size)

    num = 0
    den = 0
    for i in range(len(y)):
        if y[i] == y_hat[i] == cls:
            num+=1
        if y_hat[i] == cls:
            den+=1

    if den == 0: return None
    return round(100*num/den, 2)

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size)
    
    num = 0
    den = 0
    for i in range(len(y)):
        if y[i] == y_hat[i] == cls:
            num+=1
        if y[i] == cls:
            den+=1
    if den == 0: return None
    return round(100*num/den, 2)

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    assert(y_hat.size == y.size)
    return round((sum(i*i for i in y_hat-y)/len(y))**0.5, 4)

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert(y_hat.size == y.size)
    return round(sum(abs(i) for i in y_hat-y)/len(y), 4)

def ln_inf(y):
    return abs(max(y))

def l2_norm(y):
    ret = 0
    for i in y:
        ret+=i*i
    return (ret)**0.5